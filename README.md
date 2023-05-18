## SO Freunde

# Install poetry
~~~
curl -sSL https://install.python-poetry.org | python3 -
~~~
Add poetry to your PATH / Umgebungsvariablen \
Test if it's working
~~~
poetry --version
~~~

# Set up ssh key

- Windows need gitbash and pagent
- Linux is ready to go

Linux\
~~~
eval $("ssh-agent")
ssh-keygen -t ed25519
ssh-add .ssh/<your_key_name>
~~~
Put the public key located in `.ssh` folder into your gitlab acc \
Windows
Use the gui of pagent

# Clone the repo
~~~
git clone git@gitlab.lrz.de:00000000014A71A4/predictprotein1_solubility.git
~~~

# setup the poetry env
navigate into the folder with the `project.toml` file
~~~
poetry update
~~~

# Best practices

- Use typing for methods and arguments
- Use keyword arguments when calling methods
- Please use ssh key and git commit / push 
- use the os module when dealing with paths to make it complatible with windows/linux
- add docstrings and comments
- create branches for possibly breaking changes
