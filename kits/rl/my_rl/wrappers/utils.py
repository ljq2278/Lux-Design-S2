def recyc_dct_to_tups(recyc_dct):
    tups = []
    for gid, g_info in recyc_dct.items():
        for pid, p_info in g_info.items():
            for uid, u_info in p_info.items():
                tups.append(('__'.join([gid, pid, uid]), u_info))
    return tups


def tups_to_recyc_dct(tups):
    recyc_dct = {}
    for gid_pid_uid, u_info in tups:
        gid, pid, uid = gid_pid_uid.split('__')
        if gid not in recyc_dct.keys():
            recyc_dct[gid] = {}
        if pid not in recyc_dct[gid].keys():
            recyc_dct[gid][pid] = {}
        recyc_dct[gid][pid][uid] = u_info
    return recyc_dct
