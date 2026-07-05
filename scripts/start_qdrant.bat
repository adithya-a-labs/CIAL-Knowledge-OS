@echo off
setlocal
cd /d "%~dp0.."
docker compose -f docker-compose.qdrant.yml up -d
if errorlevel 1 exit /b %errorlevel%
docker compose -f docker-compose.qdrant.yml ps
