@echo off
setlocal

echo Preparing executable_dist directory...

:: Create executable_dist directory if it doesn't exist
if not exist "executable_dist" (
    mkdir "executable_dist"
    if %errorlevel% neq 0 (
        echo Error: Failed to create executable_dist directory.
        exit /b 1
    )
)

:: Copy EqualizerApp.exe
copy /Y "dist\EqualizerApp.exe" "executable_dist\EqualizerApp.exe"
if %errorlevel% neq 0 (
    echo Error: Failed to copy EqualizerApp.exe.
    exit /b 1
)

:: Copy icon file
copy /Y "Equalizer_30016.ico" "executable_dist\Equalizer_30016.ico"
if %errorlevel% neq 0 (
    echo Error: Failed to copy Equalizer_30016.ico.
    exit /b 1
)

echo executable_dist directory prepared successfully.
endlocal
exit /b 0