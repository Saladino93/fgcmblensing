
UCMB = 'ucmb'
LCMB = 'lcmb'
TSZ = 'tSZ'
KSZ = 'kSZ'
RADIO = 'radio'
CIB = 'cib'
GALACTIC = 'galactic dust'

COMPONENTS = [UCMB, LCMB, TSZ, KSZ, RADIO, CIB, GALACTIC]
COLORS = ['blue', 'black', 'red', 'green', 'cyan', 'orange', 'yellow']

def get_color_for_sky_component(component: str):
    return COLORS[COMPONENTS.index(component)]
