freq_tag_lut = {
    "hitting": 0,
    "kicking": 1,
    "pushing": 2,
    "grab scratch": 3,
    "head butting": 4,
    "hair pulling": 5,
    "biting": 6,
    "choking": 7,
    "sib-head banging": 8,
    "sib-head hit": 9,
    "sib-self hit": 10,
    "sib-biting": 11,
    "sib-eye poking": 12,
    "sib-body slam": 13,
    "sib-hair pulling": 14,
    "sib-choking": 15,
    "sib-pinch scratch": 16,
    "throwing obj": 17,
    "kick hit obj": 18,
    "flip furniture": 19,
    "flopping": 20
}

dur_tag_lut = {
    "st-rocking": 21,
    "st-hand flap": 22,
    "st-touch tap": 23,
    "st-head swin": 24,
    "stereo-vox": 25
}

nb_tag = 26

tag_count = 27

all_tags = {
    "hitting": 0,
    "kicking": 1,
    "pushing": 2,
    "grab scratch": 3,
    "head butting": 4,
    "hair pulling": 5,
    "biting": 6,
    "choking": 7,
    "sib-head banging": 8,
    "sib-head hit": 9,
    "sib-self hit": 10,
    "sib-biting": 11,
    "sib-eye poking": 12,
    "sib-body slam": 13,
    "sib-hair pulling": 14,
    "sib-choking": 15,
    "sib-pinch scratch": 16,
    "throwing obj": 17,
    "kick hit obj": 18,
    "flip furniture": 19,
    "flopping": 20,
    "st-rocking": 21,
    "st-hand flap": 22,
    "st-touch tap": 23,
    "st-head swin": 24,
    "stereo-vox": 25,
    "no-behavior": 26
}

class_list = ['hitting', 'throwingobj', 'nobehavior']

all_class_list = ['hitting', 'kicking', 'pushing', 'grabbingscratching', 'head butting', 'hair pull', 'biting',
                  'choking', 'SIB headbang', 'SIB headhit', 'SIB self-hit', 'SIB biting', 'SIB eyepoke',
                  'SIB body slam', 'SIB hair pull', 'SIB choking', 'SIB pinch scratch', 'throw object',
                  'kick hit object', 'flip furniture', 'flopping', 'stereoypy rocking', 'stereoypy hand flap',
                  'no pbx']

freq_classes = [
    'hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking',
    'sib-head bang', 'sib-head hit', 'sib-self-hit', 'sib-biting', 'sib-eye poke', 'sib-body slam',
    'sib-hair pull', 'sib-choking', 'sib-pinch_scratch', 'throwing object', 'kick_hit object', 'flip furniture',
    'flop', 'no pbx'
]
dur_classes = [
    'st- rock', 'st-hand flap', 'st-touch/tap', 'st-head swing', 'stereovox'
]

all_classes = [
    *freq_classes,
    # *dur_classes
]