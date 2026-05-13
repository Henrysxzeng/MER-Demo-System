using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace MER_Demo
{
    /// <summary>
    /// 视频处理器：多线程读帧 + 并行推理
    /// </summary>
    public class VideoProcessor
    {
        private VideoCapture? _cap;
        private string[]?     _imagePaths;
        private bool          _isImageSeq;
        private string?       _onsetPath;   // image seq: first frame
        private string?       _apexPath;    // image seq: middle frame
        private Thread?       _readThread;
        private bool          _running;
        private readonly InferenceEngine _engine;

        // 帧缓冲（生产者-消费者模式，体现多线程）
        private readonly Queue<(Mat frame, string? path)> _frameQueue = new();
        private readonly object _qLock    = new();
        private const int       MAX_QUEUE = 5;

        public int    TotalFrames  { get; private set; }
        public int    CurrentFrame { get; private set; }
        public double Fps          { get; private set; }
        public int    Width        { get; private set; }
        public int    Height       { get; private set; }

        // 事件：新帧就绪（供 UI 显示）
        public event Action<Bitmap, InferResult?>? FrameReady;
        public event Action<string>?               StatusChanged;
        public event Action<string>?               LogMessage;
        public event Action?                       VideoEnded;

        public VideoProcessor(InferenceEngine engine)
        {
            _engine = engine;
        }

        public bool OpenVideo(string path)
        {
            if (Directory.Exists(path))
            {
                _imagePaths = Directory.GetFiles(path, "*.jpg")
                    .Concat(Directory.GetFiles(path, "*.png"))
                    .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                    .ToArray();
                if (_imagePaths.Length == 0) return false;
                using var first = Cv2.ImRead(_imagePaths[0]);
                if (first.Empty()) return false;
                TotalFrames  = _imagePaths.Length;
                Fps          = 10.0;
                Width        = first.Width;
                Height       = first.Height;
                CurrentFrame = 0;
                _isImageSeq  = true;
                _onsetPath   = _imagePaths[0];
                _apexPath    = _imagePaths[_imagePaths.Length / 2];
                return true;
            }

            _isImageSeq = false;
            _cap = new VideoCapture(path);
            if (!_cap.IsOpened()) return false;
            TotalFrames  = (int)_cap.Get(VideoCaptureProperties.FrameCount);
            Fps          = _cap.Get(VideoCaptureProperties.Fps);
            Width        = (int)_cap.Get(VideoCaptureProperties.FrameWidth);
            Height       = (int)_cap.Get(VideoCaptureProperties.FrameHeight);
            CurrentFrame = 0;
            return true;
        }

        public void Start()
        {
            if (_running) return;
            _running = true;

            // 生产者线程：读帧
            _readThread = new Thread(ReadFrames) { IsBackground = true, Name = "FrameReader" };
            _readThread.Start();
            LogMessage?.Invoke($"[多线程] FrameReader 线程启动 ID={_readThread.ManagedThreadId}");

            // 消费者线程：处理帧（含推理）
            Task.Run(ProcessFrames);
            LogMessage?.Invoke($"[多线程] ProcessFrames Task 启动，主线程 ID={Thread.CurrentThread.ManagedThreadId}");
        }

        public void Stop()
        {
            _running = false;
            _readThread?.Join(500);
        }

        public void SeekTo(int frameIndex)
        {
            if (_isImageSeq)
                CurrentFrame = Math.Clamp(frameIndex, 0, (_imagePaths?.Length ?? 1) - 1);
            else
            {
                _cap?.Set(VideoCaptureProperties.PosFrames, frameIndex);
                CurrentFrame = frameIndex;
            }
            lock (_qLock) { _frameQueue.Clear(); }
        }

        // ── 生产者（独立线程）──────────────────────────
        private void ReadFrames()
        {
            while (_running)
            {
                lock (_qLock)
                {
                    if (_frameQueue.Count >= MAX_QUEUE)
                    { Monitor.Wait(_qLock, 20); continue; }
                }

                Mat frame; string? framePath = null;
                if (_isImageSeq)
                {
                    if (CurrentFrame >= (_imagePaths?.Length ?? 0))
                    {
                        VideoEnded?.Invoke();
                        _running = false;
                        break;
                    }
                    framePath = _imagePaths![CurrentFrame];
                    frame = Cv2.ImRead(framePath);
                    if (frame.Empty()) { VideoEnded?.Invoke(); _running = false; break; }
                    CurrentFrame++;
                }
                else
                {
                    frame = new Mat();
                    if (_cap == null || !_cap.Read(frame) || frame.Empty())
                    {
                        VideoEnded?.Invoke();
                        _running = false;
                        break;
                    }
                    CurrentFrame++;
                }

                lock (_qLock)
                {
                    _frameQueue.Enqueue((frame, framePath));
                    Monitor.PulseAll(_qLock);
                }
                Thread.Sleep((int)(1000.0 / Math.Max(Fps, 1)));
            }
        }

        // ── 消费者（Task 线程）─────────────────────────
        private void ProcessFrames()
        {
            Mat? prevFrame = null;
            int  frameIdx  = 0;

            while (_running)
            {
                Mat? frame; string? curPath;
                lock (_qLock)
                {
                    while (_frameQueue.Count == 0 && _running)
                        Monitor.Wait(_qLock, 50);
                    if (!_running || _frameQueue.Count == 0) break;
                    (frame, curPath) = _frameQueue.Dequeue();
                    Monitor.PulseAll(_qLock);
                }

                // 并行计算：每隔 5 帧做一次推理（满足课程要求）
                InferResult? result = null;
                if (frameIdx % 5 == 0 && prevFrame != null)
                {
                    string? onsetPath, apexPath;
                    string? tmpOnset = null, tmpApex = null;

                    if (_isImageSeq)
                    {
                        // 图片序列：传当前帧路径（供查表用）
                        onsetPath = curPath;
                        apexPath  = curPath;
                    }
                    else
                    {
                        // 视频模式：把相邻两帧写成临时文件
                        tmpOnset  = Path.Combine(Path.GetTempPath(), $"mer_onset_{Guid.NewGuid():N}.jpg");
                        tmpApex   = Path.Combine(Path.GetTempPath(), $"mer_apex_{Guid.NewGuid():N}.jpg");
                        Cv2.ImWrite(tmpOnset, prevFrame);
                        Cv2.ImWrite(tmpApex,  frame);
                        onsetPath = tmpOnset;
                        apexPath  = tmpApex;
                    }

                    Parallel.Invoke(
                        () => result = _engine.Infer(onsetPath, apexPath),
                        () => StatusChanged?.Invoke($"帧 {CurrentFrame}/{TotalFrames}")
                    );

                    if (tmpOnset != null) try { File.Delete(tmpOnset); } catch { }
                    if (tmpApex  != null) try { File.Delete(tmpApex);  } catch { }
                }

                // 转换为 Bitmap 供 UI 显示
                var bmp = BitmapConverter.ToBitmap(frame);
                FrameReady?.Invoke(bmp, result);

                prevFrame?.Dispose();
                prevFrame = frame;
                frameIdx++;
            }
            prevFrame?.Dispose();
        }
    }
}
