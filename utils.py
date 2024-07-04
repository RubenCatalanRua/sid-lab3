import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os

def draw_history(history, title, data_path, filename):
    data = pd.DataFrame({'Episode': range(1, len(history) + 1), title: history})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y=title, data=data)

    plt.title(title + ' Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()

    # save figure
    os.makedirs(data_path, exist_ok=True)
    plt.savefig(f"{data_path}/{filename}.png")
    plt.close()


def draw_history_2(reward_per_epoch_per_agent, title, data_path, filename):

    plt.figure(figsize=(10, 6))

    for agent_id, rewards in enumerate(reward_per_epoch_per_agent):
        plt.plot(rewards, label=f'Agente {agent_id}')

    # plt.xlabel('Episodios')
    # plt.ylabel('Recompensas acumuladas')
    # plt.title('Evolución de las recompensas acumuladas por agente')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # agent_rewards = {f"agent_{i}" for i in range(len)}
    # df = pd.DataFrame({
    #     'agent_0': arr0,
    #     'agent_1': arr1
    # })

    # plt.figure(figsize=(10, 6))
    # plt.plot(df['agent_0'], label='Agent 0')
    # plt.plot(df['agent_1'], label='Agent 1')

    # Add title and labels
    plt.title(title + ' Over Episodes By Agent')
    plt.xlabel('Episode')
    plt.ylabel(title)
    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    # save figure
    os.makedirs(data_path, exist_ok=True)
    plt.savefig(f"{data_path}/{filename}.png")
    plt.close()