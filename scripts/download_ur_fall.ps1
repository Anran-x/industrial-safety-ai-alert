$ErrorActionPreference = "Continue"
$base = "https://fenix.ur.edu.pl/~mkepski/ds/data"
$out = "D:\Study\CV_Projects\industrial_safety_ai_alert\data\fall_detection\ur_fall"
$ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

New-Item -ItemType Directory -Path "$out\falls" -Force | Out-Null
New-Item -ItemType Directory -Path "$out\adl" -Force | Out-Null

$ok = 0; $fail = 0
function Get-File($url, $dest) {
    try {
        if (Test-Path $dest) { return "skip" }
        $wc = New-Object System.Net.WebClient
        $wc.Headers.Add("User-Agent", $ua)
        $wc.DownloadFile($url, $dest)
        return "ok"
    } catch {
        if (Test-Path $dest) { Remove-Item $dest -Force }
        return "fail"
    }
}

foreach ($i in 1..30) {
    $name = "fall-{0:D2}-cam0.mp4" -f $i
    $r = Get-File "$base/$name" "$out\falls\$name"
    if ($r -eq "ok") { $ok++ } elseif ($r -eq "fail") { $fail++; Write-Host "FAIL: $name" }
    Write-Host "fall $i : $r"
}
foreach ($i in 1..40) {
    $name = "adl-{0:D2}-cam0.mp4" -f $i
    $r = Get-File "$base/$name" "$out\adl\$name"
    if ($r -eq "ok") { $ok++ } elseif ($r -eq "fail") { $fail++; Write-Host "FAIL: $name" }
    Write-Host "adl $i : $r"
}
Write-Host "=== DONE ok=$ok fail=$fail ==="
