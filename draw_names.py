import random

bree_fam = ["Dylan",
            "Emmie",
            "Wyatt"]

ry_fam = ["Adi",
          "Seth",
          "Miles"]

ty_fam = ["Scott",
          "Logan"]

fam_list = [bree_fam, ry_fam, ty_fam]
all_kiddos = bree_fam + ry_fam + ty_fam
max_iter = 100
outer_iter = 0

valid_soln = False

while not valid_soln and outer_iter < max_iter:
    still_need_gift = all_kiddos.copy()
    give_dict = {}
    for giver in all_kiddos:
        valid_pair = False
        n_iter = 0
        while not valid_pair and n_iter < max_iter:
            receiver = random.choice(still_need_gift)
            giver_fam = [giver in fam for fam in fam_list]
            receiver_fam = [receiver in fam for fam in fam_list]
            if giver_fam != receiver_fam and giver != receiver:
                valid_pair = True
                give_dict[giver] = receiver
                still_need_gift.remove(receiver)
            n_iter += 1

        if n_iter >= max_iter:
            valid_soln = False
        else:
            valid_soln = True

if valid_soln:
    [print(f"{giver} gives to {receiver}") for giver, receiver in give_dict.items()]
else:
    print("No solution possible")
