@echo off
SET p=%cd%
cd %p%
echo installing virtualenv utility...
pip install virtualenv
virtualenv env
call env\Scripts\activate.bat
pip install -r requirements.txt
echo finished set up for maria, kostas, giorgos project ...