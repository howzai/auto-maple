using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace MapleCaptureHost;

internal static class Program
{
    private const string ExactClassicTitle = "新楓之谷：經典版";

    private static readonly string[] AllowedProcessNames =
    {
        "Maplestory_Classic",
        "MapleStory",
        "MapleStoryClassic"
    };

    private static readonly string[] RejectedProcessNames =
    {
        "chrome",
        "msedge",
        "firefox",
        "opera",
        "brave",
        "iexplore",
        "ApplicationFrameHost"
    };

    [STAThread]
    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.WriteLine("MapleCaptureHost - Windows Graphics Capture backend");

        if (!OperatingSystem.IsWindowsVersionAtLeast(10, 0, 18362))
        {
            Console.Error.WriteLine("Windows 10 1903 (build 18362) or newer is required.");
            return 2;
        }

        using var shutdown = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            shutdown.Cancel();
        };

        using var output = new SharedFrameBuffer();
        if (args.Any(arg => string.Equals(arg, "--test-pattern", StringComparison.OrdinalIgnoreCase)))
        {
            Console.WriteLine("Publishing deterministic 640x360 BGRA test frames at 30 FPS.");
            Console.WriteLine("Press Ctrl+C to stop.");
            try
            {
                await TestPatternSource.RunAsync(output, shutdown.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Normal Ctrl+C shutdown.
            }
            return 0;
        }

        var window = FindTargetWindow();
        if (window.Handle == IntPtr.Zero)
        {
            Console.Error.WriteLine("Target game window was not found.");
            return 3;
        }

        Console.WriteLine(
            $"Target window found: '{window.Title}' " +
            $"process={window.ProcessName} size={window.Width}x{window.Height} " +
            $"HWND=0x{window.Handle.ToInt64():X}"
        );
        Console.WriteLine("Starting Windows Graphics Capture. Press Ctrl+C to stop.");

        try
        {
            using var capture = new GraphicsCaptureService(window.Handle, output);
            capture.Start();
            await Task.Delay(Timeout.Infinite, shutdown.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"Windows Graphics Capture failed: {exception}");
            return 4;
        }

        return 0;
    }

    private static WindowMatch FindTargetWindow()
    {
        var matches = new List<WindowMatch>();

        EnumWindows((handle, _) =>
        {
            if (!IsWindowVisible(handle) || IsIconic(handle))
            {
                return true;
            }

            var length = GetWindowTextLengthW(handle);
            if (length <= 0)
            {
                return true;
            }

            var builder = new StringBuilder(length + 1);
            _ = GetWindowTextW(handle, builder, builder.Capacity);
            var title = builder.ToString().Trim();
            if (string.IsNullOrWhiteSpace(title))
            {
                return true;
            }

            var processName = GetProcessName(handle);
            if (RejectedProcessNames.Any(name =>
                    processName.Equals(name, StringComparison.OrdinalIgnoreCase)))
            {
                return true;
            }

            if (!GetWindowRect(handle, out var rect))
            {
                return true;
            }

            var width = Math.Max(0, rect.Right - rect.Left);
            var height = Math.Max(0, rect.Bottom - rect.Top);

            // Reject splash windows, launchers, thumbnails and transitional windows.
            if (width < 800 || height < 500)
            {
                return true;
            }

            var exactTitle = title.Equals(ExactClassicTitle, StringComparison.OrdinalIgnoreCase);
            var mapleProcess = AllowedProcessNames.Any(name =>
                processName.Equals(name, StringComparison.OrdinalIgnoreCase));

            // A valid target must either have the exact game title or be a known
            // MapleStory executable whose title starts with the game name.
            if (!exactTitle && !(mapleProcess && title.StartsWith("新楓之谷", StringComparison.OrdinalIgnoreCase)))
            {
                return true;
            }

            var score = 0;
            if (exactTitle)
            {
                score += 1000;
            }
            if (mapleProcess)
            {
                score += 2000;
            }
            score += Math.Min(width * height / 1000, 1500);

            matches.Add(new WindowMatch(handle, title, processName, width, height, score));
            return true;
        }, IntPtr.Zero);

        return matches
            .OrderByDescending(match => match.Score)
            .ThenByDescending(match => match.Width * (long)match.Height)
            .FirstOrDefault();
    }

    private static string GetProcessName(IntPtr handle)
    {
        _ = GetWindowThreadProcessId(handle, out var processId);
        if (processId == 0)
        {
            return string.Empty;
        }

        try
        {
            using var process = Process.GetProcessById((int)processId);
            return process.ProcessName;
        }
        catch
        {
            return string.Empty;
        }
    }

    private readonly record struct WindowMatch(
        IntPtr Handle,
        string Title,
        string ProcessName,
        int Width,
        int Height,
        int Score
    );

    private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool IsIconic(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowTextLengthW(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowTextW(IntPtr hWnd, StringBuilder text, int maxCount);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetWindowRect(IntPtr hWnd, out Rect rect);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [StructLayout(LayoutKind.Sequential)]
    private struct Rect
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
}
