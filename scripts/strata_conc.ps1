param(
  [int]$N = 4,
  [int]$Tokens = 200,
  [string]$Label = ''
)
# ai3080 Strata 并发基准：N 路同时发请求，看聚合吞吐。
# 用法: .\strata_conc.ps1 -N 4 -Tokens 200 -Label "A-4cards"
$u = 'http://<LAN-IP>:8080/v1/chat/completions'
$sw = [Diagnostics.Stopwatch]::StartNew()
$res = 1..$N | ForEach-Object -Parallel {
  $u2 = 'http://<LAN-IP>:8080/v1/chat/completions'
  $body = @{
    model       = 'x'
    temperature = 0
    max_tokens  = $using:Tokens
    messages    = @(@{ role = 'user'; content = "Write a plain 150-word description of a lighthouse at night, variation $_." })
  } | ConvertTo-Json -Depth 5 -Compress
  $r = Invoke-RestMethod -Uri $u2 -Method Post -Headers @{ 'Content-Type' = 'application/json' } -Body $body -TimeoutSec 900
  [pscustomobject]@{ ct = $r.usage.completion_tokens; pt = $r.usage.prompt_tokens }
} -ThrottleLimit $N
$sw.Stop()
$sum = ($res | Measure-Object -Property ct -Sum).Sum
$wall = [math]::Round($sw.Elapsed.TotalSeconds, 2)
$agg = [math]::Round($sum / $sw.Elapsed.TotalSeconds, 1)
"$Label concurrent N=$N tokens=$Tokens : total_completion=$sum wall=${wall}s aggregate=$agg tok/s"
