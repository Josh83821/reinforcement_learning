from random import random, randint, choices

def generate_loss_list():

    loss_list = []

    for i in range(100):
        loss_list.append(random())

    return loss_list


def generate_rewards_list():

    rewards_list = []

    for i in range(100):
        rewards_list.append(randint(50, 500))

    return rewards_list


def gen_word(length):
    alph = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    string_list = choices(alph, k=length)

    string = ''
    for letter in string_list:
        string += letter
    return string

def gen_model_alg_data():

    model_name = gen_word(5)
    alg_name = gen_word(5)

    return_dict = {
    'model_name' : model_name,
    'alg_name' : alg_name
    }

    return return_dict
    
def gen_data():

    loss_list = generate_loss_list()
    rewards_list = generate_rewards_list()
    alg_data = gen_model_alg_data()

    return loss_list, rewards_list, alg_data
    
