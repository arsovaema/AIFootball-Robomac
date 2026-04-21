# AIFootball
Python script for 3v3 football game used in Robomac competition as part of the AI category.

## Rules
* 3 vs 3 Players football game, everyone is a player and a goalkeeper at the same time
* Match duration is 90 seconds with 45 seconds of each halftime
* `AIFootball.py` is the official simulation script
* Team_name folder contains the .py script and files that need to be modified by each team
* You define the `team_properties()` function  in `Manager.py` to declare your team and player names
* Use `decision()` function to gather game information and control your players
* You can remove the render to speed up training

## Every player has

|mass        | player mass            |
|:-----------|:-----------------------|
|radius      | player size            |
|acceleration| player top acceleration|
|speed       | player top speed       |
|shoot power | player top shoot power |

* Heavier players can push out lighter players 
* Players with higher radius take up more space
* Ball travels further when shot by higher power shooters

## Your task
* Knowing the position of the ball and all the players on the field, calculate your next move
* Act with force on your players, the simulation script will take care of the position and the velocity
* If you want to shoot the ball then set the appropirate variable `shot_request` to `True`. The ball will be shot on the next collision between the player and the ball

## Good luck
