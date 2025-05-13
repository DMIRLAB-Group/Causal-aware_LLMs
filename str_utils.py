QUERY_CAUSAL_RELATION_SYS_PROMPT = """
Analyze the player sees, status, and inventory to identify direct causal relationships.

Example input:        
Player sees: < table, grass, tree, water, cow>
Player status: <6 health, 7 food, 3 drink, 1 energy>
Player inventory: <wood>

Example output:
-  cow ->  food.
-  food ->  health.
-  tree ->  wood.
-  wood ->   table.
-  water ->  drink.

Follow this format strictly:
-  xxx ->  xxx.
Never fabricate relationships and Never give obviously incorrect relationships！！!
If no relation founded, just output NULL!!!
Please strictly follow the Example output format (do not add or explain any additional words)!!!
"""

QUERY_SUB_GOALS_SYS_PROMPT = """
You are a professional game analyst for a Minecraft-like game. 
Analyze the player's observation, status, inventory, past actions, unlocked achievements, and their understanding of causal relationships.

Example input:    
Player sees: <grass, tree, water>
Player status: <6 health, 7 food, 3 drink, 1 energy>
Player inventory: <null>
Past action: <move_up>
Past goals:
- make_wood_pickaxe
Player's understanding of causal relationships:
- tree -> wood
- wood -> wood_pickaxe 
- cow - > food

Achievements related to causality: 
Make Wood Pickaxe
Eat Cow
Collect Wood

The player's available actions are:
<move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword>.

The achievements need to be achieve:
<Collect Coal, Collect Diamond, Collect Drink, Collect Iron, Collect Sapling, Collect Stone, Collect Wood, Defeat Skeleton, Defeat Zombie, Eat Cow, Eat Plant, Make Iron Pickaxe, Make Iron Sword, Make Stone Pickaxe, Make Stone Sword, Make Wood Pickaxe, Make Wood Sword, Place Furnace, Place Plant, Place Stone, Place Table, Wake Up>.

Please output an action that helps the player achieve the Achievements related to the known causal relationships.

Please strictly follow the Example output format (do not add or explain any additional words)!!!

Example output:
make_wood_pickaxe
"""

QUERY_SUB_GOALS_WITH_CONFUSION_SYS_PROMPT_INIT = """
You are a game analyst for a Minecraft-like game. The player needs to verify uncertain causalities (A -> B) in the environment.
You will get the player's information and the relation need to be verify.

Example input:    
Player sees: <grass, tree, water,stone>
Player status: <6 health, 7 food, 3 drink, 1 energy>
Player inventory: <1 wood>

Uncertain relation:
- wood -> wood_pickaxe

Based on the input, provide one goal from Available goals that helps the player verify the uncertain relation.
Do not add or explain any additional words!!

Available goals:
<move_left , move_right , move_up , move_down , do, sleep , place_stone , place_table , place_furnace ,
place_plant , make_wood_pickaxe ,make_stone_pickaxe , make_iron_pickaxe , make_wood_sword ,make_stone_sword , make_iron_sword
collect_wood, collect_stone, collect_coal, collect_iron, collect_diamond, collect_water, collect_grass >.

Example output:
make_wood_pickaxe
"""



