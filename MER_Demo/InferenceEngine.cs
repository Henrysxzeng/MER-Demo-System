using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace MER_Demo
{
    /// <summary>
    /// 与 Python 推理后端通信的引擎（独立进程，stdin/stdout JSON）
    /// </summary>
    public class InferenceEngine : IDisposable
    {
        private Process? _proc;
        private StreamWriter? _writer;
        private StreamReader? _reader;
        private readonly object _lock = new();
        public bool IsReady { get; private set; }

        private const string PYTHON = @"E:\Anaconda\envs\hpl_htnet\python.exe";
        private const string SCRIPT = @"D:\表情识别\MER_Demo_System\backend\inference_server.py";

        public event Action<string>? LogMessage;

        public bool Start()
        {
            try
            {
                var utf8 = new System.Text.UTF8Encoding(false);
                var psi = new ProcessStartInfo(PYTHON, $"\"{SCRIPT}\"")
                {
                    RedirectStandardInput   = true,
                    RedirectStandardOutput  = true,
                    RedirectStandardError   = true,
                    UseShellExecute         = false,
                    CreateNoWindow          = true,
                    StandardInputEncoding   = utf8,
                    StandardOutputEncoding  = utf8,
                };
                _proc = Process.Start(psi) ?? throw new Exception("Failed to start Python");
                _writer = _proc.StandardInput;
                _reader = _proc.StandardOutput;
                _proc.BeginErrorReadLine();
                _proc.ErrorDataReceived += (_, e) =>
                {
                    if (e.Data != null) LogMessage?.Invoke($"[Backend] {e.Data}");
                };

                // 测试连通性
                var resp = SendCommand(new { cmd = "ping" });
                IsReady = resp?["status"]?.ToString() == "ok";
                return IsReady;
            }
            catch (Exception ex)
            {
                LogMessage?.Invoke($"[Engine] Start failed: {ex.Message}");
                return false;
            }
        }

        public bool LoadModel(string weightsDir)
        {
            var resp = SendCommand(new { cmd = "load_model", weights_dir = weightsDir });
            bool ok = resp?["status"]?.ToString() == "ok";
            string msg = resp?["message"]?.ToString() ?? "(no message)";
            LogMessage?.Invoke($"[LoadModel] {msg}");
            return ok;
        }

        public InferResult? Infer(string? onsetPath = null, string? apexPath = null,
                                  int[] ? faceRect = null)
        {
            var req = new
            {
                cmd       = "infer",
                onset     = onsetPath,
                apex      = apexPath,
                face_rect = faceRect ?? new[] { 0, 0, 100, 100 },
            };
            var resp = SendCommand(req);
            if (resp?["status"]?.ToString() != "ok") return null;

            return new InferResult
            {
                Prediction  = resp["prediction"]?.ToObject<int>() ?? 0,
                LabelEn     = resp["label_en"]?.ToString() ?? "",
                LabelCn     = resp["label_cn"]?.ToString() ?? "",
                Color       = resp["color"]?.ToString() ?? "#888888",
                Negative    = resp["probabilities"]?["Negative"]?.ToObject<double>() ?? 0,
                Positive    = resp["probabilities"]?["Positive"]?.ToObject<double>() ?? 0,
                Surprise    = resp["probabilities"]?["Surprise"]?.ToObject<double>() ?? 0,
            };
        }

        private JObject? SendCommand(object request)
        {
            lock (_lock)
            {
                try
                {
                    if (_writer == null || _reader == null) return null;
                    string json = JsonConvert.SerializeObject(request);
                    _writer.WriteLine(json);
                    _writer.Flush();
                    string? line = _reader.ReadLine();
                    return line != null ? JObject.Parse(line) : null;
                }
                catch (Exception ex)
                {
                    LogMessage?.Invoke($"[Engine] SendCommand error: {ex.Message}");
                    return null;
                }
            }
        }

        public void Dispose()
        {
            try { SendCommand(new { cmd = "quit" }); } catch { }
            _proc?.Kill();
            _proc?.Dispose();
        }
    }

    public class InferResult
    {
        public int    Prediction  { get; set; }
        public string LabelEn     { get; set; } = "";
        public string LabelCn     { get; set; } = "";
        public string Color       { get; set; } = "#888888";
        public double Negative    { get; set; }
        public double Positive    { get; set; }
        public double Surprise    { get; set; }
    }
}
