[CmdletBinding()]
param(
    [Parameter()][string]$OutputDirectory = "dist",
    [Parameter()][switch]$IncludeData
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repository
try {
    $commit = (git rev-parse HEAD).Trim()
    if ($commit -notmatch '^[0-9a-f]{40}$') { throw "Git HEAD is not an exact commit" }
    if (git status --porcelain --untracked-files=all) {
        throw "The repository must be clean before creating an upload package"
    }
    $output = Join-Path $repository $OutputDirectory
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $output = (Resolve-Path $output).Path
    $stagingParent = Join-Path $output ("modern-upload-stage-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $stagingParent | Out-Null
    $stagingParent = (Resolve-Path $stagingParent).Path
    $payload = Join-Path $stagingParent "payload"
    New-Item -ItemType Directory -Path $payload | Out-Null
    $payload = (Resolve-Path $payload).Path
    if (-not $payload.StartsWith($stagingParent + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing staging path outside the newly created parent"
    }

    $sourceTar = Join-Path $stagingParent "source.tar"
    if ($IncludeData) {
        git archive --format=tar --output=$sourceTar $commit
    }
    else {
        git archive --format=tar --output=$sourceTar $commit -- . ':(exclude)Oxford Dataset'
    }
    if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
    tar -xf $sourceTar -C $payload
    git bundle create (Join-Path $payload "repository.bundle") HEAD
    if ($LASTEXITCODE -ne 0) { throw "git bundle failed" }

    $sssd = Join-Path $stagingParent "sssd"
    git clone --quiet https://github.com/AI4HealthUOL/SSSD.git $sssd
    git -C $sssd checkout --quiet --detach 4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed
    if ((git -C $sssd rev-parse HEAD).Trim() -ne "4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed") {
        throw "SSSD checkout does not match the pinned commit"
    }
    $sssdDestination = Join-Path $payload "third_party\sssd\source"
    New-Item -ItemType Directory -Force -Path $sssdDestination | Out-Null
    git -C $sssd archive --format=tar --output=(Join-Path $stagingParent "sssd.tar") HEAD
    tar -xf (Join-Path $stagingParent "sssd.tar") -C $sssdDestination

    if ($IncludeData -and -not (Test-Path -LiteralPath (Join-Path $payload "Oxford Dataset") -PathType Container)) {
        throw "Oxford Dataset was not included in the exact Git archive"
    }
    @"
#!/usr/bin/env bash
set -Eeuo pipefail
git init -q
git bundle verify repository.bundle
git fetch -q repository.bundle $commit
git reset --hard FETCH_HEAD
test "`$(git rev-parse HEAD)" = "$commit"
echo "Upload restored at $commit. Next: bash scripts/run_modern_imputation_matpool.sh prepare"
"@ | Set-Content -LiteralPath (Join-Path $payload "bootstrap.sh") -Encoding utf8NoBOM

    $archive = Join-Path $output ("modern-imputation-upload-" + $commit + ".tar.gz")
    if (Test-Path -LiteralPath $archive) { throw "Upload archive already exists: $archive" }
    tar -czf $archive -C $payload .
    if ($LASTEXITCODE -ne 0) { throw "tar creation failed" }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    "$hash  $([IO.Path]::GetFileName($archive))" | Set-Content -LiteralPath ($archive + ".sha256") -Encoding ascii
    Write-Output $archive
}
finally {
    Pop-Location
    if ($stagingParent -and (Test-Path -LiteralPath $stagingParent)) {
        $resolvedStage = (Resolve-Path -LiteralPath $stagingParent).Path
        $resolvedOutput = (Resolve-Path -LiteralPath $output).Path
        if (-not $resolvedStage.StartsWith($resolvedOutput + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove staging directory outside output"
        }
        Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    }
}
