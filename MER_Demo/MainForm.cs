using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Threading;
using System.Windows.Forms;

namespace MER_Demo
{
    public partial class MainForm : Form
    {
        private readonly InferenceEngine _engine  = new();
        private VideoProcessor?          _video;
        private InferResult?             _lastResult;
        private readonly System.Windows.Forms.Timer _uiTimer = new();

        // ── 布局常量 ──────────────────────────────────────
        private const int PAD = 10;

        public MainForm()
        {
            Text            = "微表情识别演示系统 — CAST";
            Size            = new Size(1100, 720);
            MinimumSize     = new Size(900, 600);
            BackColor       = Color.FromArgb(18, 18, 30);
            Font            = new Font("Microsoft YaHei", 9f);
            StartPosition   = FormStartPosition.CenterScreen;

            BuildUI();
            HookEvents();

            _uiTimer.Interval = 33; // ~30fps UI 刷新
            _uiTimer.Tick    += UiTimer_Tick;
        }

        // ──────────────── 控件声明 ────────────────────────
        private Panel     _videoPanel   = null!;
        private PictureBox _videoBox    = null!;
        private Panel     _infoPanel    = null!;
        private Label     _lblTitle     = null!;
        private Label     _lblEmotion   = null!;
        private Label     _lblEmotionCn = null!;
        private Panel     _barNeg       = null!;
        private Panel     _barPos       = null!;
        private Panel     _barSur       = null!;
        private Label     _lblNeg       = null!;
        private Label     _lblPos       = null!;
        private Label     _lblSur       = null!;
        private Label     _lblStatus    = null!;
        private TrackBar  _seekBar      = null!;
        private Button    _btnOpen      = null!;
        private Button    _btnOpenSeq   = null!;
        private Button    _btnPlay      = null!;
        private Button    _btnStop      = null!;
        private RichTextBox _logBox     = null!;
        private bool      _playing;

        private void BuildUI()
        {
            // ── 顶部工具栏 ──────────────────────────────────
            var toolbar = new Panel
            {
                Dock = DockStyle.Top, Height = 44,
                BackColor = Color.FromArgb(28, 28, 45),
                Padding = new Padding(PAD, 6, PAD, 6)
            };
            _btnOpen    = MakeBtn("📂 打开视频",   Color.FromArgb(52, 73, 94));
            _btnOpenSeq = MakeBtn("🖼 图片序列",   Color.FromArgb(41, 128, 185));
            _btnPlay    = MakeBtn("▶ 播放",        Color.FromArgb(39, 174, 96));
            _btnPlay.Enabled = false;
            _btnStop    = MakeBtn("■ 停止",        Color.FromArgb(192, 57, 43));
            _btnStop.Enabled = false;
            var btnLoad = MakeBtn("🤖 加载模型",   Color.FromArgb(142, 68, 173));

            _btnOpen.Click    += BtnOpen_Click;
            _btnOpenSeq.Click += BtnOpenSeq_Click;
            _btnPlay.Click    += BtnPlay_Click;
            _btnStop.Click    += BtnStop_Click;
            btnLoad.Click     += BtnLoad_Click;

            toolbar.Controls.AddRange(new Control[]
                { _btnOpen, _btnOpenSeq, _btnPlay, _btnStop, btnLoad });
            int x = PAD;
            foreach (Control c in toolbar.Controls)
            { c.Left = x; x += c.Width + 8; }

            // ── 视频区域（左） ──────────────────────────────
            _videoPanel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.Black,
                Padding = new Padding(PAD)
            };
            _videoBox = new PictureBox
            {
                Dock = DockStyle.Fill,
                SizeMode = PictureBoxSizeMode.Zoom,
                BackColor = Color.Black
            };
            _videoPanel.Controls.Add(_videoBox);

            // ── 进度条 ──────────────────────────────────────
            var seekPanel = new Panel
            {
                Dock = DockStyle.Bottom, Height = 30,
                BackColor = Color.FromArgb(28, 28, 45)
            };
            _seekBar = new TrackBar
            {
                Dock = DockStyle.Fill, Minimum = 0, Maximum = 100,
                TickStyle = TickStyle.None, Height = 30
            };
            _seekBar.MouseUp += SeekBar_MouseUp;
            seekPanel.Controls.Add(_seekBar);

            // ── 状态栏 ──────────────────────────────────────
            _lblStatus = new Label
            {
                Dock = DockStyle.Bottom, Height = 22,
                ForeColor = Color.FromArgb(160, 160, 180),
                BackColor = Color.FromArgb(22, 22, 38),
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(PAD, 0, 0, 0),
                Text = "就绪 — 请打开视频文件"
            };

            var leftPanel = new Panel { Dock = DockStyle.Fill };
            leftPanel.Controls.Add(_videoPanel);
            leftPanel.Controls.Add(seekPanel);
            leftPanel.Controls.Add(_lblStatus);

            // ── 右侧信息面板 ────────────────────────────────
            _infoPanel = new Panel
            {
                Dock = DockStyle.Right, Width = 280,
                BackColor = Color.FromArgb(22, 22, 38),
                Padding = new Padding(PAD)
            };
            BuildInfoPanel();

            // ── 日志（底部可选） ────────────────────────────
            _logBox = new RichTextBox
            {
                Dock = DockStyle.Bottom, Height = 90,
                BackColor = Color.FromArgb(12, 12, 20),
                ForeColor = Color.FromArgb(100, 200, 100),
                BorderStyle = BorderStyle.None,
                ReadOnly = true, ScrollBars = RichTextBoxScrollBars.Vertical,
                Font = new Font("Consolas", 8f)
            };

            Controls.Add(leftPanel);
            Controls.Add(_infoPanel);
            Controls.Add(_logBox);
            Controls.Add(_lblStatus);
            Controls.Add(toolbar);
        }

        private void BuildInfoPanel()
        {
            int y = PAD;

            _lblTitle = MakeLabel("微表情识别系统", 13, FontStyle.Bold);
            _lblTitle.Top = y; y += 30;

            var sep1 = MakeSep(y); y += 16;

            var lblEmotionHdr = MakeLabel("当前情感状态", 9, FontStyle.Regular,
                Color.FromArgb(120, 120, 160));
            lblEmotionHdr.Top = y; y += 22;

            _lblEmotion = MakeLabel("---", 28, FontStyle.Bold, Color.White);
            _lblEmotion.Top = y; y += 40;

            _lblEmotionCn = MakeLabel("等待推理...", 12, FontStyle.Regular,
                Color.FromArgb(160, 160, 200));
            _lblEmotionCn.Top = y; y += 30;

            var sep2 = MakeSep(y); y += 16;

            var lblProbHdr = MakeLabel("置信度分布", 9, FontStyle.Regular,
                Color.FromArgb(120, 120, 160));
            lblProbHdr.Top = y; y += 22;

            // 三个情感的进度条
            (_barNeg, _lblNeg) = MakeBar("负面 Negative", Color.FromArgb(231, 76, 60),  ref y);
            (_barPos, _lblPos) = MakeBar("正面 Positive", Color.FromArgb(39, 174, 96),  ref y);
            (_barSur, _lblSur) = MakeBar("惊讶 Surprise", Color.FromArgb(41, 128, 185), ref y);

            y += 10;
            var sep3 = MakeSep(y); y += 16;

            // CAST 模型说明
            var lblInfo = new Label
            {
                Text = "模型: CAST\n(CBAM-Augmented Semantic Transformer)\n\nCAS(ME)³: 71.45% UF1 (+7.33%)\nMEGC2019: 86.35% UF1 (+7.9%)",
                Top = y, Left = PAD, Width = _infoPanel.Width - PAD*2,
                Height = 90, ForeColor = Color.FromArgb(120, 140, 170),
                Font = new Font("Microsoft YaHei", 8f)
            };

            _infoPanel.Controls.AddRange(new Control[]
            {
                _lblTitle, sep1, lblEmotionHdr, _lblEmotion, _lblEmotionCn,
                sep2, lblProbHdr, _barNeg, _lblNeg, _barPos, _lblPos,
                _barSur, _lblSur, sep3, lblInfo
            });
        }

        // ── 辅助控件构建 ──────────────────────────────────
        private Label MakeLabel(string text, float size, FontStyle style,
            Color? color = null)
        {
            return new Label
            {
                Text = text, Left = PAD, Width = _infoPanel.Width - PAD*2,
                AutoSize = false, Height = (int)(size * 2.2),
                Font = new Font("Microsoft YaHei", size, style),
                ForeColor = color ?? Color.White
            };
        }

        private Panel MakeSep(int y)
        {
            return new Panel
            {
                Top = y, Left = PAD, Width = _infoPanel.Width - PAD*2,
                Height = 1, BackColor = Color.FromArgb(50, 50, 70)
            };
        }

        private (Panel bar, Label lbl) MakeBar(string name, Color color, ref int y)
        {
            var lbl = new Label
            {
                Text = $"{name}  0%", Top = y, Left = PAD,
                Width = _infoPanel.Width - PAD*2, Height = 18,
                ForeColor = color, Font = new Font("Microsoft YaHei", 8f)
            };
            y += 20;
            var bg = new Panel
            {
                Top = y, Left = PAD, Width = _infoPanel.Width - PAD*2,
                Height = 10, BackColor = Color.FromArgb(40, 40, 60)
            };
            var bar = new Panel
            {
                Top = 0, Left = 0, Width = 0, Height = 10,
                BackColor = color
            };
            bg.Controls.Add(bar);
            y += 18;
            return (bar, lbl);
        }

        private static Button MakeBtn(string text, Color back)
        {
            return new Button
            {
                Text = text, Width = 110, Height = 30,
                BackColor = back, ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat, Font = new Font("Microsoft YaHei", 8.5f),
                Cursor = Cursors.Hand
            };
        }

        // ── 事件处理 ──────────────────────────────────────
        private void HookEvents()
        {
            _engine.LogMessage += msg => AppendLog(msg);

            // 键盘快捷键（满足课程要求）
            KeyPreview = true;
            KeyDown += (_, e) =>
            {
                if (e.KeyCode == Keys.Space)
                {
                    e.SuppressKeyPress = true;
                    if (_playing) { BtnStop_Click(null, null!); AppendLog("[Key] Space → 暂停"); }
                    else          { BtnPlay_Click(null, null!); AppendLog("[Key] Space → 播放"); }
                }
                if (e.KeyCode == Keys.O && e.Control) { BtnOpen_Click(null, null!); AppendLog("[Key] Ctrl+O → 打开文件"); }
                if (e.KeyCode == Keys.Escape)         { Stop();                     AppendLog("[Key] Esc → 停止"); }
            };
        }

        private void BtnOpen_Click(object? s, EventArgs e)
        {
            using var dlg = new OpenFileDialog
            {
                Filter = "视频文件|*.avi;*.mp4;*.mov;*.mkv|所有文件|*.*",
                Title = "打开微表情视频"
            };
            if (dlg.ShowDialog() != DialogResult.OK) return;
            OpenPath(dlg.FileName);
        }

        private void BtnOpenSeq_Click(object? s, EventArgs e)
        {
            using var dlg = new FolderBrowserDialog
            {
                Description = "选择图片序列文件夹（含 jpg/png 帧）",
                UseDescriptionForTitle = true
            };
            if (dlg.ShowDialog() != DialogResult.OK) return;
            OpenPath(dlg.SelectedPath);
        }

        private void OpenPath(string path)
        {
            Stop();
            _video = new VideoProcessor(_engine);
            _video.FrameReady    += Video_FrameReady;
            _video.StatusChanged += msg => SetStatus(msg);
            _video.LogMessage    += msg => AppendLog(msg);
            _video.VideoEnded    += () => Invoke(() =>
            {
                _playing = false; _btnPlay.Enabled = true; _btnStop.Enabled = false;
                SetStatus("播放完毕");
            });

            if (!_video.OpenVideo(path))
            { MessageBox.Show("无法打开视频/图片序列", "错误"); return; }

            _seekBar.Maximum = Math.Max(1, _video.TotalFrames);
            _btnPlay.Enabled = true;
            string name = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar));
            SetStatus($"已加载: {name}  ({_video.Width}×{_video.Height}, {_video.Fps:F1}fps, {_video.TotalFrames}帧)");
            AppendLog($"[Open] {path}");
        }

        private void BtnLoad_Click(object? s, EventArgs e)
        {
            using var dlg = new FolderBrowserDialog { Description = "选择 CAST 权重文件夹" };
            if (dlg.ShowDialog() != DialogResult.OK) return;

            SetStatus("正在加载模型，请稍候...");
            var dir = dlg.SelectedPath;
            var t = new Thread(() =>
            {
                if (!_engine.IsReady) _engine.Start();
                bool ok = _engine.LoadModel(dir);
                SetStatus(ok ? "✅ 模型已加载，可以开始推理" : "❌ 模型加载失败");
                AppendLog($"[Model] {(ok ? "OK" : "FAIL")} {dir}");
            }) { IsBackground = true };
            t.Start();
        }

        private void BtnPlay_Click(object? s, EventArgs e)
        {
            if (_video == null) return;
            _video.Start();
            _playing = true;
            _uiTimer.Start();
            _btnPlay.Enabled = false;
            _btnStop.Enabled = true;
        }

        private void BtnStop_Click(object? s, EventArgs e) => Stop();

        private void Stop()
        {
            _video?.Stop();
            _uiTimer.Stop();
            _playing = false;
            _btnPlay.Enabled = _video != null;
            _btnStop.Enabled = false;
        }

        private void SeekBar_MouseUp(object? s, MouseEventArgs e)
        {
            _video?.SeekTo(_seekBar.Value);
        }

        // ── 帧就绪回调（非 UI 线程） ──────────────────────
        private void Video_FrameReady(Bitmap bmp, InferResult? result)
        {
            if (InvokeRequired)
            {
                BeginInvoke(() => Video_FrameReady(bmp, result));
                return;
            }
            _videoBox.Image?.Dispose();
            _videoBox.Image = bmp;

            if (result != null) _lastResult = result;

            if (_video != null)
                _seekBar.Value = Math.Min(_seekBar.Maximum, _video.CurrentFrame);
        }

        // ── UI 定时器：更新情感显示 ───────────────────────
        private void UiTimer_Tick(object? s, EventArgs e)
        {
            if (_lastResult == null) return;
            var r = _lastResult;

            var c = ColorTranslator.FromHtml(r.Color);
            _lblEmotion.Text      = r.LabelEn;
            _lblEmotion.ForeColor = c;
            _lblEmotionCn.Text    = r.LabelCn;

            int bw = _infoPanel.Width - PAD * 2 - 4;
            UpdateBar(_barNeg, _lblNeg, "负面 Negative", r.Negative, bw);
            UpdateBar(_barPos, _lblPos, "正面 Positive", r.Positive, bw);
            UpdateBar(_barSur, _lblSur, "惊讶 Surprise", r.Surprise, bw);
        }

        private static void UpdateBar(Panel bar, Label lbl,
            string name, double prob, int maxW)
        {
            bar.Width = (int)(maxW * prob);
            int pct = (int)(prob * 100);
            lbl.Text = $"{name}  {pct}%";
        }

        // ── 工具方法 ──────────────────────────────────────
        private void SetStatus(string msg) =>
            Invoke(() => _lblStatus.Text = msg);

        private void AppendLog(string msg) =>
            Invoke(() =>
            {
                _logBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {msg}\n");
                _logBox.ScrollToCaret();
            });

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            Stop();
            _engine.Dispose();
            base.OnFormClosing(e);
        }
    }
}
