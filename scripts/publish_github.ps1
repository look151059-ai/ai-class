Param(
  # You can either pass -RepoSlug "owner/repo" or pass -Owner + -RepoName.
  [string]$RepoSlug,

  [string]$Owner = "look151059",

  [string]$RepoName = "ai-class",

  [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

function Assert-Git {
  $git = Get-Command git -ErrorAction SilentlyContinue
  if (-not $git) {
    Write-Host "找不到 git 指令。請先完成 Git for Windows 安裝，並重新開啟 VS Code/終端機。" -ForegroundColor Yellow
    Write-Host "安裝後驗證：" -ForegroundColor Yellow
    Write-Host "  git --version" -ForegroundColor Yellow
    exit 1
  }
}

Assert-Git

$repoOwner = $Owner
$repoName = $RepoName
if ($RepoSlug) {
  $parts = $RepoSlug.Split("/", 2)
  if ($parts.Length -ne 2 -or (-not $parts[0]) -or (-not $parts[1])) {
    throw "RepoSlug 格式錯誤，請用 owner/repo，例如 look151059/ai-class"
  }
  $repoOwner = $parts[0]
  $repoName = $parts[1]
}

if (-not (Test-Path ".git")) {
  git init | Out-Null
}

git checkout -B $Branch | Out-Null

# Basic identity (only set if missing)
$name = git config --get user.name
if (-not $name) { git config user.name $repoOwner | Out-Null }
$email = git config --get user.email
if (-not $email) { git config user.email "$repoOwner@users.noreply.github.com" | Out-Null }

git add -A

$hasCommit = $true
try { git rev-parse --verify HEAD | Out-Null } catch { $hasCommit = $false }
if (-not $hasCommit) {
  git commit -m "Initial commit" | Out-Null
} else {
  # Create a commit only if there are staged changes
  $staged = git diff --cached --name-only
  if ($staged) { git commit -m "Update" | Out-Null }
}

$remoteUrl = "https://github.com/$repoOwner/$repoName.git"
$hasOrigin = $true
try { git remote get-url origin | Out-Null } catch { $hasOrigin = $false }
if (-not $hasOrigin) {
  git remote add origin $remoteUrl
} else {
  git remote set-url origin $remoteUrl
}

Write-Host ""
Write-Host "準備 push 到：" -ForegroundColor Cyan
Write-Host "  $remoteUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "如果尚未在 GitHub 建立同名 repo，請先到 GitHub 建立 $repoOwner/$repoName（public/private 都可）。" -ForegroundColor Yellow
Write-Host "接著這一步會跳出 GitHub 登入/授權（Git Credential Manager）。" -ForegroundColor Yellow
Write-Host ""

git push -u origin $Branch
