$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (!(Test-Path results/latest.txt)) { throw 'No successful run. Run run.cmd first.' }
$dest = (Get-Content results/latest.txt -Raw).Trim()
& python -m pip install -r scripts/requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python package installation failed' }
& python scripts/analyze.py --plots $dest
if ($LASTEXITCODE -ne 0) { throw 'Plot generation failed' }
& python scripts/make_report.py --results $dest --output "$dest/report.docx"
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
Compress-Archive -Path "$dest/*" -DestinationPath "$dest.zip" -Force
Write-Host "REPORT SUCCESS: $dest/report.docx"
Write-Host "Updated archive: $dest.zip"
