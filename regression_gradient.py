# RL training
import mjai
from weights_bot import WeightsBot
import numpy as np
from zipfile import ZipFile
import json
import logging

logging.basicConfig(filename="regression_gradient.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger("regression gradient")
logger.setLevel(logging.DEBUG)


with open("weights_bot.py", "r") as f:
    orig_bot_str = f.read()

bot_str = "\n\nWeightsBot({0}, player_id=int(sys.argv[1])).start()"

epochs = 20
n_traj = 1
n_samples = 20
lr = 0.1

weights = np.ones(10)
submissions = ["weightsbot.zip",
               "baseline/rulebase.zip",
               "baseline/rulebase.zip",
               "baseline/rulebase.zip"]

for _ in range(epochs):
    with open('bot.py', 'w') as f:
        f.write(orig_bot_str)
        f.write(bot_str.format(np.array2string(weights, separator=",")))
    with ZipFile('weightsbot.zip', 'w') as zip:
        zip.write('bot.py')
    #Rollout n_traj trajectories, get utility
    u = 0.0
    for traj in range(n_traj):
        try:
            mjai.Simulator(submissions, logs_dir="./logs").run()
        except Exception as e:
            logger.debug("Failed to run simulator...")
            logger.debug(e)
            continue
        with open("logs/summary.json", "r") as f:
            summary = json.load(f)["kyoku"]
            for kyoku in summary:
                if kyoku["error_info"] is not None:
                    logger.debug("ERROR")
                    raise KeyboardInterrupt
            final_score = summary[-1]["end_kyoku_scores"][0]
            u += final_score - 25000
    u /= n_traj
    logger.info(u) # For sanity check. utility should be increasing.

    perturbations = [np.random.randn(10) for _ in range(n_samples)]
    perturbations = [delta/np.linalg.norm(delta) for delta in perturbations]

    delta_u = [-u for _ in range(n_samples)]
    for i, perturbation in enumerate(perturbations):
        new_weights = weights + perturbation
        with open('bot.py', 'w') as f:
            f.write(orig_bot_str)
            f.write(bot_str.format(np.array2string(new_weights, separator=",")))
        with ZipFile('weightsbot.zip', 'w') as zip:
            zip.write('bot.py')
        #Rollout n_traj trajectories, get utility
        u_i = 0.0
        for traj in range(n_traj):
            try:
                mjai.Simulator(submissions, logs_dir="./logs").run()
            except Exception as e:
                logger.debug("Failed to run simulator...")
                logger.debug(e)
                continue 
            with open("logs/summary.json", "r") as f:
                summary = json.load(f)["kyoku"]
                for kyoku in summary:
                    if kyoku["error_info"] is not None:
                        logger.debug("ERROR")
                        raise KeyboardInterrupt
                final_score = summary[-1]["end_kyoku_scores"][0]
                u_i += final_score - 25000
        u_i /= n_traj
        delta_u[i] += u_i
    perturbations = np.stack(perturbations)
    if np.linalg.norm(delta_u) > 0:
        delta_u = delta_u / np.linalg.norm(delta_u)
        nabla_u = np.dot(np.linalg.pinv(perturbations), np.array(delta_u))
        weights += lr*nabla_u
        logger.info("delta_u: "+str(delta_u))
        logger.info("nabla_u: "+str(nabla_u))

logger.info("final weights: "+str(weights))
np.save("weights", weights)