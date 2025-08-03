# PowerShell script to run local evaluation

# Check if required packages are installed
Write-Host "Checking required packages..."
$packages = @("pandas")
foreach ($package in $packages) {
    try {
        python -c "import $package" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installing $package..."
            pip install $package
        } else {
            Write-Host "$package is already installed."
        }
    } catch {
        Write-Host "Installing $package..."
        pip install $package
    }
}

# Run the evaluation script
Write-Host "`nRunning local evaluation..."
python langsmith_evaluation.py