[CmdletBinding()]
param(
    [string]$OutputDirectory = "transfer",
    [ValidateRange(1, 19)]
    [int]$CompressionLevel = 10,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputDirectory))
}

$packageStem = "lnn-imputation-data-minimal-v1"
$tarPath = Join-Path $outputRoot "$packageStem.tar"
$archivePath = "$tarPath.zst"
$checksumPath = "$archivePath.sha256"
$manifestPath = Join-Path $outputRoot "$packageStem.manifest.json"

$sources = @(
    "Oxford Dataset",
    "datasets/processed/euroc_mav",
    "datasets/raw/idol/archives/building1.zip",
    "datasets/raw/idol/archives/building2.zip",
    "datasets/raw/idol/archives/building3.zip",
    "datasets/manifests/external_datasets.json"
)

foreach ($relativePath in $sources) {
    $absolutePath = Join-Path $repositoryRoot $relativePath
    if (-not (Test-Path -LiteralPath $absolutePath)) {
        throw "Required bundle input is missing: $absolutePath"
    }
}

$tar = Get-Command tar -ErrorAction Stop
$zstd = Get-Command zstd -ErrorAction Stop
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$generatedPaths = @($tarPath, $archivePath, $checksumPath, $manifestPath)
$existingPaths = @($generatedPaths | Where-Object { Test-Path -LiteralPath $_ })
if ($existingPaths.Count -gt 0 -and -not $Force) {
    throw "Generated outputs already exist. Re-run with -Force after reviewing them: $($existingPaths -join ', ')"
}
if ($Force) {
    foreach ($path in $generatedPaths) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }
}

$sourceRows = foreach ($relativePath in $sources) {
    $absolutePath = Join-Path $repositoryRoot $relativePath
    $item = Get-Item -LiteralPath $absolutePath
    $bytes = if ($item.PSIsContainer) {
        (Get-ChildItem -LiteralPath $absolutePath -Recurse -File |
            Measure-Object -Property Length -Sum).Sum
    } else {
        $item.Length
    }
    [ordered]@{
        path = $relativePath.Replace("\", "/")
        bytes = [int64]$bytes
    }
}

Push-Location $repositoryRoot
try {
    & $tar.Source -cf $tarPath @sources
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

& $zstd.Source -T0 "-$CompressionLevel" -f $tarPath -o $archivePath
if ($LASTEXITCODE -ne 0) {
    throw "zstd failed with exit code $LASTEXITCODE"
}

Remove-Item -LiteralPath $tarPath -Force

$archive = Get-Item -LiteralPath $archivePath
$sha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$checksumLine = "$sha256  $($archive.Name)`n"
[System.IO.File]::WriteAllText(
    $checksumPath,
    $checksumLine,
    [System.Text.Encoding]::ASCII
)

$manifest = [ordered]@{
    schema_version = 1
    package = $archive.Name
    archive_format = "tar.zst"
    archive_bytes = [int64]$archive.Length
    archive_sha256 = $sha256
    created_utc = [DateTime]::UtcNow.ToString("o")
    source_bytes = [int64](($sourceRows | ForEach-Object { $_.bytes } | Measure-Object -Sum).Sum)
    sources = @($sourceRows)
    expected_recordings = [ordered]@{
        oxiod = 45
        euroc_mav = 6
        idol = 130
    }
    server_initialization = @(
        "Extract this archive from the repository root.",
        "Unzip building1.zip, building2.zip, and building3.zip into datasets/raw/idol/archives/.",
        "Do not require the full EuRoC camera or ROS-bag payloads."
    )
}
[System.IO.File]::WriteAllText(
    $manifestPath,
    (($manifest | ConvertTo-Json -Depth 8) + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)

[PSCustomObject]@{
    Archive = $archivePath
    SizeGB = [math]::Round($archive.Length / 1GB, 3)
    SHA256 = $sha256
    Checksum = $checksumPath
    Manifest = $manifestPath
}
