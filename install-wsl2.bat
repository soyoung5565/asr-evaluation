@echo off
setlocal enabledelayedexpansion
set "LOG=D:\asr\wsl-install.log"
echo [%date% %time%] batch started >> "%LOG%"

REM check admin via net session
net session >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [%date% %time%] not admin, requesting elevation >> "%LOG%"
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
echo [%date% %time%] running as ADMIN >> "%LOG%"

echo ============================================
echo  Installing WSL2 + Ubuntu (takes a few min)
echo  Window will NOT close until done.
echo  Result is logged to D:\asr\wsl-install.log
echo ============================================

echo [%date% %time%] wsl --install -d Ubuntu --no-launch >> "%LOG%"
wsl.exe --install -d Ubuntu --no-launch >> "%LOG%" 2>&1
set "RC=!ERRORLEVEL!"
echo [%date% %time%] wsl install exit code = !RC! >> "%LOG%"

if not "!RC!"=="0" goto :fail

echo.
echo  WSL2 + Ubuntu installed OK (exit code 0).
echo  A REBOOT is required to finish setup.
choice /C YN /M "Reboot now? (Y=reboot now / N=later)"
if errorlevel 2 goto :noreboot
shutdown /r /t 5
echo Rebooting in 5s...
exit /b

:fail
echo.
echo  INSTALL FAILED (exit code !RC!). See D:\asr\wsl-install.log
echo  Common causes: Windows update pending, no internet, or
echo  virtualization disabled in BIOS (Hyper-V / SVM).
pause
exit /b

:noreboot
echo OK, remember to reboot manually, then say "restarted" to Hermes.
pause
exit /b
