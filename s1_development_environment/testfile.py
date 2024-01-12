import sys #import
print(sys.executable) # Location of the Python.exe used to run this code

########################################
### HOW TO RESET VIRTUAL ENVIRONMENT ###
########################################

# Step 1: Select global interpreter in statusbar at the bottom of VS Code
# Step 2: Run the following commands in PowerShell (VS Code terminal)

# pip freeze > requirements.txt
# Remove-Item -Recurse -Force .venv

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# python -m venv .venv
# .\.venv\Scripts\Activate
# pip install -r requirements.txt




####################################
### HOW TO SET GITHUB REPOSITORY ###
####################################

# Step 1: Create new git repo
# Step 2: Run the following commands (change url)

# git remote add origin https://github.com/JonatanRasmussen/02476-Machine-Learning-Operations.git
# git branch -M main
# git push -u origin main
