import numpy as np


def create_tag_inputs(Nppl, Nproj, D, seed=1234, manipulate_proj=False):
    step_range = 30
    ppl_rate = 0.2
    proj_rate = 0.6
    np.random.seed(seed)
    # 人员标签，随机生成
    v = np.random.uniform(0, 1, (Nppl, D))
    ppl_tag = (v < ppl_rate).astype(int)

    # 项目标签，随机产生或者由人员标签生产
    if not manipulate_proj:
        w = np.random.uniform(0, 1, (Nproj, D))
        proj_tag = (w < proj_rate).astype(int)
        solution = None
    else:
        ppl_select = np.arange(Nppl)
        np.random.shuffle(ppl_select)
        solution = np.zeros((Nppl, ), dtype=int)
        proj_tag = []
        for i in range(Nproj):
            t = np.zeros((D, ), dtype=int)
            for k in range(i*3, i*3+3):
                t = t + ppl_tag[ppl_select[k], :]
                solution[ppl_select[k]] = i + 1
            proj_tag.append(t)
        proj_tag = np.array(proj_tag)

    # 项目来到时间，项目时长
    arrival = [[] for _ in range(step_range)]
    proj_length = np.zeros((Nproj, ))
    for i in range(Nproj):
        t = np.random.randint(0, step_range-1)
        dur = np.random.randint(2, 4)
        arrival[t].append(i)
        proj_length[i] = dur

    ppl_occupation = np.zeros((Nppl, step_range + 4))

    return {
        "people_tag": np.array(ppl_tag),
        "project_tag": np.array(proj_tag),
        "project_arrival": arrival,
        "proj_length": proj_length,
        "people_occupation": ppl_occupation,
        "solution": solution,
    }