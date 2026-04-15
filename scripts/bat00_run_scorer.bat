@echo off
echo ============================================
echo   comma.ai Auth Scorer - bat00 RTX 2070 Super
echo   %date% %time%
echo ============================================
echo.
echo Dependencies install to WSL native filesystem (fast).
echo Results saved to C:\Users\adpena\pact_eval\results\
echo.
echo Starting WSL...
wsl -e bash -c "bash /mnt/c/Users/adpena/pact_eval/setup.sh; echo; echo DONE. PRESS ENTER TO CLOSE; read"
pause
