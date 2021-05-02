# VirxEAI

## Requirements

+ [Windows 10](https://www.microsoft.com/en-us/software-download/windows10)
+ [Python 3.7](https://www.python.org/downloads/release/python-379/)
+ [Rocket League](https://www.rocketleague.com/) (either on Steam or Epic)
+ [BakkesMod](https://www.bakkesmod.com/)

## Installing

+ `git clone --recurse-submodules https://github.com/VirxEC/VirxEAI.git`
+ `cd VirxEAI`
+ `python -m venv env`
+ `call env\Scripts\activate.bat`
+ `pip install -r requirements.txt`

## Training

Open Rocket League and set everything to minimum (including the resolution!) and the framerate to uncapped.

+ `call env\Scripts\activate.bat`
+ `pip install -U -r requirements.txt`
+ `python main.py`

## Running normally

Either you can just do normal game speed (via the BakkesMod console) or use [RLGym's](https://rlgym.github.io/index.html) integration with [RLBot](http://rlbot.org/).
