import random
import argparse
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu  # For inferential statistics (optional)


# ------------------------------------------------------------------
# RYFF CELLS (special cells that fill one dimension)
# ------------------------------------------------------------------
RYFF_DIMENSION_CELLS = [
    ("Autonomy",               1.0, [7.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("Environmental Mastery",  1.0, [0.0, 7.0, 0.0, 0.0, 0.0, 0.0]),
    ("Personal Growth",        1.0, [0.0, 0.0, 7.0, 0.0, 0.0, 0.0]),
    ("Positive Relations",     1.0, [0.0, 0.0, 0.0, 7.0, 0.0, 0.0]),
    ("Purpose",                1.0, [0.0, 0.0, 0.0, 0.0, 7.0, 0.0]), 
    ("Self-Acceptance",        1.0, [0.0, 0.0, 0.0, 0.0, 0.0, 7.0])
]
RYFF_NAMES = [x[0] for x in RYFF_DIMENSION_CELLS]

# ------------------------------------------------------------------
# SUPER STIMULI (used only in mixed environments)
# ------------------------------------------------------------------
SUPER_STIMULI = [
    # ("Doomscroll", 5.0, [0.0, -5, -2, 0.0, -2, -5]),
    # ("Outrage Media", 15.0, [0.0, 0.0, 0.0, 0.0, 0.025, 0.0]),
    ("Outrage Media", 15.0, [-4.0, -4.0, -4.0, -4.0, 6.0, -4.0])
]

# ------------------------------------------------------------------
# SUPER STIMULI MAPPINGS (ensuring super-stumli serves as proxy for Ryff dimension)
# ------------------------------------------------------------------
SUPER_STIMULI_MAPPING = {
    "Outrage Media": "Purpose"
}

# ------------------------------------------------------------------
# ICON MAPPINGS
# ------------------------------------------------------------------
icon_paths = {
    "Autonomy": "icons/autonomy.png",
    "Environmental Mastery": "icons/env_mastery.png",
    "Personal Growth": "icons/growth.png",
    "Positive Relations": "icons/pos_relations.png",
    "Purpose": "icons/purpose.png",
    "Self-Acceptance": "icons/selfacceptance.png"
}
superstimuli_icon_paths = {
    "Doomscroll": "icons/doomscroll.png",
    "Outrage Media": "icons/outrage.png"
}

# ------------------------------------------------------------------
# UTILITY FUNCTION: Clamp
# ------------------------------------------------------------------
def clamp(val, mn=0, mx=100):
    return max(mn, min(mx, val))

# ------------------------------------------------------------------
# CELL CLASS
# ------------------------------------------------------------------
class Cell:
    def __init__(self, cell_type=None, salience=0.0, replenish_effects=None):
        self.type = cell_type if cell_type else "undefined"
        self.salience = float(salience)
        if replenish_effects is None:
            replenish_effects = [0.0]*6
        elif len(replenish_effects) != 6:
            raise ValueError("replenish_effects must have length 6.")
        self.replenish_effects = list(replenish_effects)
    def __repr__(self):
        return f"Cell(type={self.type}, salience={self.salience}, effects={self.replenish_effects})"

# ------------------------------------------------------------------
# ENVIRONMENT CREATION
# ------------------------------------------------------------------
def create_environment(grid_size, env_type="mixed"):
    rows, cols = grid_size
    env = [[None for _ in range(cols)] for _ in range(rows)]
    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(all_positions)
    
    # Place 6 special Ryff-dimension cells.
    special_positions = all_positions[:6]
    for i, (r, c) in enumerate(special_positions):
        nm, sal, eff = RYFF_DIMENSION_CELLS[i]
        env[r][c] = Cell(nm, sal, eff)
    
    remain = all_positions[6:]
    if env_type == "mixed":
        # Place exactly one instance per superstimuli mapping.
        remaining_positions = remain.copy()
        for st_type in SUPER_STIMULI_MAPPING.keys():
            if remaining_positions:
                pos = random.choice(remaining_positions)
                remaining_positions.remove(pos)
                st_tuple = next((s for s in SUPER_STIMULI if s[0] == st_type), None)
                if st_tuple:
                    r, c = pos
                    env[r][c] = Cell(st_tuple[0], st_tuple[1], st_tuple[2])
        # Fill any remaining positions with mundane cells.
        for pos in remaining_positions:
            r, c = pos
            env[r][c] = Cell("mundane", 0.0, [0.0]*6)
    elif env_type == "nourishing":
        for pos in remain:
            r, c = pos
            env[r][c] = Cell("mundane", 0.0, [0.0]*6)
    else:
        # Default to mixed behavior.
        for pos in remain:
            r, c = pos
            env[r][c] = Cell("mundane", 0.0, [0.0]*6)
    return env

# ------------------------------------------------------------------
# AGENT FUNCTIONS
# ------------------------------------------------------------------
def initialize_agent():
    return {
        'x': 0,
        'y': 0,
        'ryff_levels': [50.0]*6,
        'discovered_dims': set(),
        'discovery_score': 0,
        'mode': 'discover',
        'current_target_dim': None,
        'saturation_counts': [0]*6,
        'saturated_flags': [False]*6
    }

# ------------------------------------------------------------------
# DECAY AND CELL EFFECTS
# ------------------------------------------------------------------
def apply_decay(agent, skip_dim=None, decay_rate=0.5):
    """
    Decays all dimensions except for skip_dim.
    This is how the agent's well-being naturally decreases over time.
    """
    for i in range(6):
        if i == skip_dim:
            continue
        agent['ryff_levels'][i] = clamp(agent['ryff_levels'][i] - decay_rate)

def apply_cell_effects(agent, cell):
    """
    Increase each Ryff dimension based on the cell's effects.
    """
    for i in range(6):
        agent['ryff_levels'][i] = clamp(agent['ryff_levels'][i] + cell.replenish_effects[i])

def check_discovery(agent):
    """
    If a special cell is visited and its corresponding dimension reaches 100,
    mark that dimension as discovered.
    """
    for i in range(6):
        if agent['ryff_levels'][i] >= 100 and i not in agent['discovered_dims']:
            agent['discovered_dims'].add(i)
            agent['discovery_score'] += 1

# ------------------------------------------------------------------
# MOVEMENT
# ------------------------------------------------------------------
def move_one_step(agent, target):
    if not target:
        return
    rx, ry = agent['x'], agent['y']
    tx, ty = target
    if rx < tx:
        agent['x'] += 1
    elif rx > tx:
        agent['x'] -= 1
    elif ry < ty:
        agent['y'] += 1
    elif ry > ty:
        agent['y'] -= 1

def find_best_cell_for_dim(env, d):
    rows, cols = len(env), len(env[0])
    best_score = -float('inf')
    best_coord = None
    for r in range(rows):
        for c in range(cols):
            cell = env[r][c]
            # emphasize immediate positive reward and salience, discount long-term negatives
            positive_effect = max(cell.replenish_effects[d], 0)
            score = positive_effect * cell.salience
            if score > best_score and positive_effect > 0:
                best_score = score
                best_coord = (r, c)
    return best_coord

# ------------------------------------------------------------------
# HELPER: Clear Target if Dimension is Full
# ------------------------------------------------------------------
def clear_target_if_full(agent):
    d = agent.get('current_target_dim')
    if d is not None and agent['ryff_levels'][d] >= 100:
        agent['current_target_dim'] = None

# ------------------------------------------------------------------
# STAY-IF-FILLING: Should the agent remain on the current cell?
# ------------------------------------------------------------------
def stay_if_filling(agent, environment):
    d = agent['current_target_dim']
    if d is None:
        return False
    # Only remain if the cell offers a positive effect for that dimension
    # and that dimension is not yet 100.
    if agent['ryff_levels'][d] >= 100:
        return False
    cell = environment[agent['x']][agent['y']]
    return cell.replenish_effects[d] > 0

# ------------------------------------------------------------------
# DISCOVER MODE POLICY
# ------------------------------------------------------------------
def discover_mode_policy(agent, environment):
    # If the agent is on a special cell, force the target dimension.
    cell = environment[agent['x']][agent['y']]
    if cell.type in RYFF_NAMES:
        agent['current_target_dim'] = RYFF_NAMES.index(cell.type)
        if agent['ryff_levels'][agent['current_target_dim']] < 100:
        # Stay here until filled.
            return
    # If on a superstimuli cell, set the target dimension based on mapping.
    elif cell.type in SUPER_STIMULI_MAPPING:
        target_dim_name = SUPER_STIMULI_MAPPING[cell.type]
        agent['current_target_dim'] = RYFF_NAMES.index(target_dim_name)
        if agent['ryff_levels'][agent['current_target_dim']] < 100:
            return  # Stay here until filled.

    clear_target_if_full(agent)
    if agent['current_target_dim'] is not None and stay_if_filling(agent, environment):
        return
    if agent['current_target_dim'] is None:
        undiscovered = [i for i in range(6) if i not in agent['discovered_dims']]
        if undiscovered:
            agent['current_target_dim'] = random.choice(undiscovered)
        else:
            directions = [(1,0), (-1,0), (0,1), (0,-1)]
            ch = random.choice(directions)
            agent['x'] = clamp(agent['x'] + ch[0], 0, len(environment)-1)
            agent['y'] = clamp(agent['y'] + ch[1], 0, len(environment[0])-1)
            return
    td = agent['current_target_dim']
    if agent['ryff_levels'][td] < 100:
        bc = find_best_cell_for_dim(environment, td)
        move_one_step(agent, bc)

# ------------------------------------------------------------------
# REPLENISH MODE POLICY
# ------------------------------------------------------------------
def replenish_mode_policy(agent, environment, repl_thr=40):
    clear_target_if_full(agent)
    if stay_if_filling(agent, environment):
        return
    if agent['current_target_dim'] is None:
        dims_below = [(i, agent['ryff_levels'][i]) for i in range(6) if agent['ryff_levels'][i] < repl_thr]
        if dims_below:
            dd, lvl = min(dims_below, key=lambda x: x[1])
            agent['current_target_dim'] = dd
        else:
            dims_below_100 = [(i, agent['ryff_levels'][i]) for i in range(6) if agent['ryff_levels'][i] < 100]
            if dims_below_100:
                dd2, lvl2 = min(dims_below_100, key=lambda x: x[1])
                agent['current_target_dim'] = dd2
            else:
                directions = [(1,0), (-1,0), (0,1), (0,-1)]
                ch = random.choice(directions)
                rx = clamp(agent['x'] + ch[0], 0, len(environment)-1)
                ry = clamp(agent['y'] + ch[1], 0, len(environment[0])-1)
                agent['x'], agent['y'] = rx, ry
                return
    td = agent['current_target_dim']
    if td is not None and agent['ryff_levels'][td] < 100:
        bc = find_best_cell_for_dim(environment, td)
        move_one_step(agent, bc)

# ------------------------------------------------------------------
# EXPLORE MODE POLICY
# ------------------------------------------------------------------
def explore_mode_policy(agent, environment):
    cell = environment[agent['x']][agent['y']]
    
    # Stay on current cell if replenishing dimension below 100
    if agent['current_target_dim'] is not None and stay_if_filling(agent, environment):
        return
    
    # If agent just arrived at a special Ryff or superstimuli cell, set the target dimension
    if cell.type in RYFF_NAMES:
        agent['current_target_dim'] = RYFF_NAMES.index(cell.type)
        if agent['ryff_levels'][agent['current_target_dim']] < 100:
            return  # Stay until filled
    elif cell.type in SUPER_STIMULI_MAPPING:
        target_name = SUPER_STIMULI_MAPPING[cell.type]
        agent['current_target_dim'] = RYFF_NAMES.index(target_name)
        if agent['ryff_levels'][agent['current_target_dim']] < 100:
            return  # Stay until filled
    
    # If no compelling reason to stay, perform salience-weighted exploration
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    possible_moves = []
    salience_weights = []
    
    rows, cols = len(environment), len(environment[0])
    
    for dx, dy in directions:
        nx, ny = clamp(agent['x'] + dx, 0, rows - 1), clamp(agent['y'] + dy, 0, cols - 1)
        target_cell = environment[nx][ny]
        possible_moves.append((nx, ny))
        # Ensure salience is positive and nonzero to guide movement effectively
        sal_weight = max(0.1, target_cell.salience)
        salience_weights.append(sal_weight)

    # Normalize weights into probabilities
    total_weight = sum(salience_weights)
    probabilities = [w / total_weight for w in salience_weights]
    
    # Choose the next move based on salience-weighted probabilities
    next_move = random.choices(possible_moves, weights=probabilities, k=1)[0]
    agent['x'], agent['y'] = next_move
    agent['current_target_dim'] = None  # Clear target dimension after moving


# ------------------------------------------------------------------
# MODE UPDATE
# ------------------------------------------------------------------
def update_agent_mode(agent, disc_thr=3, repl_thr=40, forced_mode=None):
    if forced_mode:
        agent['mode'] = forced_mode
        return
    for i in range(6):
        if agent['ryff_levels'][i] < repl_thr:
            agent['mode'] = 'replenish'
            return
    if agent['discovery_score'] < disc_thr:
        agent['mode'] = 'discover'
        return
    # If not all dimensions have been discovered (< 6 discovered)
    elif agent['discovery_score'] < 6:
        # The probability to choose discover mode decreases as discovery_score increases.
        p_discover = (6 - agent['discovery_score']) / 6.0
        if random.random() < p_discover:
            agent['mode'] = 'discover'
        else:
            agent['mode'] = 'explore'
    agent['mode'] = 'explore'

# ------------------------------------------------------------------
# MAIN AGENT STEP
# ------------------------------------------------------------------
def agent_step(agent, env, disc_thr=3, repl_thr=40, forced_mode=None, decay_rate=0.5, debug=False):
    update_agent_mode(agent, disc_thr, repl_thr, forced_mode=forced_mode)
    old_levels = agent['ryff_levels'][:]
    if agent['mode'] == 'discover':
        discover_mode_policy(agent, env)
    elif agent['mode'] == 'replenish':
        replenish_mode_policy(agent, env, repl_thr)
    else:
        explore_mode_policy(agent, env)
    r, c = agent['x'], agent['y']
    cell = env[r][c]
    apply_cell_effects(agent, cell)
    check_discovery(agent)
    skip_d = agent.get('current_target_dim')
    apply_decay(agent, skip_dim=skip_d, decay_rate=decay_rate)
    if debug:
        print(f"[DEBUG] Mode={agent['mode']} TargetDim={agent.get('current_target_dim')}, Levels={agent['ryff_levels']}, Discovery={agent['discovery_score']}")

# ------------------------------------------------------------------
# VISUALIZATION COLOR MAPPING
# ------------------------------------------------------------------
def get_color(cell):
    cdict = {
        "Autonomy": "green",
        "Environmental Mastery": "green",
        "Personal Growth": "green",
        "Positive Relations": "green",
        "Purpose": "green",
        "Self-Acceptance": "green",
        "mundane": "lightgray",
        "Outrage Media": "red",
    }
    return cdict.get(cell.type, "white")

# ------------------------------------------------------------------
# ANIMATION FUNCTION WITH ICON OVERLAYS
# ------------------------------------------------------------------
def animate_simulation(env, agent, steps=200, disc_thr=3, repl_thr=40,
                       decay_rate=0.5, forced_mode=None, env_type="mixed",
                       save_mp4=None, debug_print=False):
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    rows, cols = len(env), len(env[0])
    cdict = {"green": 0, "red": 1, "lightgray": 2, "white": 3}
    cmap = mcolors.ListedColormap([
        "xkcd:light seafoam green","xkcd:muted pink","xkcd:cool grey","white"
    ])
    
    def env2grid():
        gr = []
        for r in range(rows):
            row = []
            for c in range(cols):
                col = get_color(env[r][c])
                row.append(cdict.get(col,3))
            gr.append(row)
        return gr
    
    init_grid = env2grid()
    im_env = ax1.imshow(init_grid, cmap=cmap, vmin=0, vmax=3)
    scat_agent = ax1.scatter(agent['y'], agent['x'], marker='s', color='black', s=100)
    ax1.set_title("Grid")
    
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    for r in range(rows):
        for c in range(cols):
            cell = env[r][c]
            if cell.type in RYFF_NAMES:
                try:
                    img = plt.imread(icon_paths[cell.type])
                    oi = OffsetImage(img, zoom=0.5)
                    ab = AnnotationBbox(oi, (c + 0.0, r + 0.0), frameon=False)
                    ax1.add_artist(ab)
                except Exception as e:
                    print(f"Error loading icon for {cell.type} at ({r}, {c}): {e}")
            elif env_type=="mixed" and cell.type in superstimuli_icon_paths:
                try:
                    img = plt.imread(superstimuli_icon_paths[cell.type])
                    oi = OffsetImage(img, zoom=0.5)
                    ab = AnnotationBbox(oi, (c + 0.0, r + 0.0), frameon=False)
                    ax1.add_artist(ab)
                except Exception as e:
                    print(f"Error loading superstimuli icon for {cell.type} at ({r}, {c}): {e}")
    
    xinds = range(6)
    bars = ax2.bar(xinds, agent['ryff_levels'], color='xkcd:lavender pink', alpha=0.6)
    ax2.set_ylim([0,100])
    ax2.set_xticks(xinds)
    ax2.set_xticklabels(["Autonomy","Environmental \nMastery\n","Growth","Positive \nRelations\n","Purpose","Self \nAcceptance\n"], rotation=30)
    ax2.set_ylabel("Ryff Level")
    ax2.set_title("Ryff Levels")
    
    stepcount = [0]
    def init():
        im_env.set_data(init_grid)
        return [im_env, scat_agent] + list(bars)
    
    def update(frame):
        stepcount[0] += 1
        step_id = stepcount[0]
        agent_step(agent, env, disc_thr, repl_thr, forced_mode, decay_rate, debug=debug_print)
        newg = env2grid()
        im_env.set_data(newg)
        scat_agent.set_offsets([[agent['y'], agent['x']]])
        for i, b in enumerate(bars):
            lvl = agent['ryff_levels'][i]
            b.set_height(lvl)
            if lvl < repl_thr:
                b.set_color("red")
            else:
                b.set_color("xkcd:lavender pink")
        ax1.set_title(f"Grid Step {step_id} - Mode={agent['mode']} - TDim={agent['current_target_dim']}")
        ax2.set_title(f"Ryff Levels (Disc={agent['discovery_score']}/6)")
        return [im_env, scat_agent] + list(bars)
    
    anim = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=500, repeat=False)
    # plt.tight_layout(rect=[0, 0.10, 0, 0])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    if save_mp4:
        writer = FFMpegWriter(fps=5, bitrate=1800)
        anim.save(save_mp4, writer=writer)
        print("Saved to", save_mp4)
    plt.show()
    return anim

# ------------------------------------------------------------------
# EVALUATION FUNCTIONS
# ------------------------------------------------------------------
def run_simulation_trial(env_type, num_steps, decay_rate, disc_thr, repl_thr, forced_mode=None, seed=None, save_video=False, trial_id=None, total_trials=None):
    if seed is not None:
        random.seed(seed)
    env = create_environment((8, 8), env_type)
    agent = initialize_agent()
    # time_superstimuli = 0
    # time_replenishing = 0
    superstimuli_saturations = 0
    ryff_saturations = 0

    # If save_video is True and this is the last trial, run the simulation with video saving.
    if save_video and (total_trials is not None and trial_id == total_trials - 1):
        animate_simulation(env, agent,
                           steps=num_steps,
                           disc_thr=disc_thr,
                           repl_thr=repl_thr,
                           decay_rate=decay_rate,
                           forced_mode=forced_mode,
                           env_type=env_type,
                           save_mp4=f"simulation_trial_{trial_id}_{env_type}" + (f"_{forced_mode}" if forced_mode else "") + ".mp4",
                           debug_print=False)
    else:
        for step in range(num_steps):
            prev_levels = agent['ryff_levels'][:]  
            agent_step(agent, env, disc_thr, repl_thr, forced_mode=forced_mode, decay_rate=decay_rate)
            cell = env[agent['x']][agent['y']]
            # If it’s a superstimulus cell, see if the relevant dimension got newly saturated
            if cell.type in SUPER_STIMULI_MAPPING:
                time_superstimuli += 1
                # e.g. if Outrage Media maps to "Purpose"
                mapped_dim = RYFF_NAMES.index(SUPER_STIMULI_MAPPING[cell.type])
                if prev_levels[mapped_dim] < 100 and agent['ryff_levels'][mapped_dim] >= 100:
                    superstimuli_saturations += 1
            
            # Else if it’s one of the six Ryff dimension cells
            elif cell.type in RYFF_NAMES:
                time_replenishing += 1
                mapped_dim = RYFF_NAMES.index(cell.type)
                if prev_levels[mapped_dim] < 100 and agent['ryff_levels'][mapped_dim] >= 100:
                    ryff_saturations += 1
    
    return {
        'environment': env_type,
        'forced_mode': forced_mode,
        'Purpose': agent['ryff_levels'][RYFF_NAMES.index("Purpose")],
        'Autonomy': agent['ryff_levels'][RYFF_NAMES.index("Autonomy")],
        'Personal Growth': agent['ryff_levels'][RYFF_NAMES.index("Personal Growth")],
        'Environmental Mastery': agent['ryff_levels'][RYFF_NAMES.index("Environmental Mastery")],
        'Positive Relations': agent['ryff_levels'][RYFF_NAMES.index("Positive Relations")],
        'Self-Acceptance': agent['ryff_levels'][RYFF_NAMES.index("Self-Acceptance")],
        'time_superstimuli': time_superstimuli,
        'time_replenishing': time_replenishing,
        'superstimuli_saturations': superstimuli_saturations,
        'ryff_saturations': ryff_saturations,
    }

def evaluate_simulations(env_type, num_trials, num_steps, decay_rate, disc_thr, repl_thr, forced_mode=None):
    data = []
    for trial in range(num_trials):
        # Save video only for the last trial.
        save_video = (trial == num_trials - 1)
        # If you still want to compare both environment types in each trial, loop over them:
        for et in ['mixed', 'nourishing']:
            result = run_simulation_trial(et, num_steps, decay_rate, disc_thr, repl_thr, forced_mode=forced_mode, seed=trial,
                                          save_video=None, trial_id=trial, total_trials=num_trials)
            result['trial'] = trial
            data.append(result)
    df = pd.DataFrame(data)
    # Build a filename dynamically based on env_type and forced_mode.
    filename = f"simulation_results_high_cost"
    if forced_mode:
        filename += f"_{forced_mode}"
    filename += ".csv"
    
    df.to_csv(filename, index=False)
    print(f"Simulation results saved to {filename}")
    return df


# ------------------------------------------------------------------
# MAIN EXECUTION WITH PARAMETERS
# ------------------------------------------------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Gridworld Simulation with Multi-Mode Agent and Evaluation.")
    parser.add_argument("--decay_rate", type=float, default=0.5, help="Global decay each step.")
    parser.add_argument("--discovery_thr", type=int, default=3)
    parser.add_argument("--replenish_thr", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--forced_mode", type=str, default=None, help="Force agent mode: discover, replenish, explore.") 
    parser.add_argument("--env_type", type=str, default="mixed", choices=["mixed", "nourishing"])
    parser.add_argument("--save_mp4", type=str, default=False, help="Save animation as MP4.")
    parser.add_argument("--debug_print", action="store_true", help="Print debug lines each step to confirm net effect.")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation (multiple trials) instead of animation.")
    parser.add_argument("--num_trials", type=int, default=30, help="Number of simulation trials for evaluation.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generator for reproducibility.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.evaluate:
        # Run evaluation trials and save results to a CSV file.
        df_all = evaluate_simulations(args.env_type, args.num_trials, args.num_steps, args.decay_rate, args.discovery_thr, args.replenish_thr, forced_mode=args.forced_mode)
    else:
        env = create_environment((8, 8), args.env_type)
        agent = initialize_agent()
        animate_simulation(env, agent,
                           steps=args.num_steps,
                           disc_thr=args.discovery_thr,
                           repl_thr=args.replenish_thr,
                           decay_rate=args.decay_rate,
                           forced_mode=args.forced_mode,
                           env_type=args.env_type,
                           save_mp4=args.save_mp4,
                           debug_print=args.debug_print)
