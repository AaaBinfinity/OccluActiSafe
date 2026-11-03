# Windows任务计划程序的独立宿主；只执行用户批准的一次N17恢复，不自动重试。
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $OutputEncoding
$evidence = 'H:\OccluActiSafeDataset\artifacts\节点资料\N17_检测器比较\03_验收证据\n17_lr_contrast_delivery_20261103\进程中断恢复_20261103'
$repository = Split-Path -Parent $PSScriptRoot
# 独占声明先于任何日志重定向；任务结束后误点重启也不会覆盖旧证据。
$claim = Join-Path $evidence '独立恢复队列_启动声明.json'
$claimValue = [ordered]@{time=(Get-Date).ToString('o'); process_id=$PID; automatic_retry=$false} | ConvertTo-Json
New-Item -ItemType File -Path $claim -Value $claimValue -ErrorAction Stop | Out-Null
$exitCode = 1
$failure = $null
try {
    Set-Location -LiteralPath $repository
    & (Join-Path $repository '.venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'recover_n17_lr_contrast.py') --action run *> (Join-Path $evidence '独立恢复队列_输出与错误.txt')
    $exitCode = $LASTEXITCODE
}
catch {
    $failure = $_.Exception.Message
}
finally {
    [ordered]@{
        time = (Get-Date).ToString('o')
        exit_code = $exitCode
        error = $failure
        completed = ($exitCode -eq 0)
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence '独立恢复队列_退出.json') -Encoding utf8
}
exit $exitCode
