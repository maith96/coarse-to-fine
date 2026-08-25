"""Randomise the passage's facts, consistently, one assignment per document.

Every training answer in the fixed-fact corpus is memorisable, so memorisation
is a sufficient strategy for the whole training set and gradient descent never
has to learn to read. This removes that option: each document gets its own cast,
colours, times and quantities, so no constant answer survives across documents
and the only thing that predicts the answer is the narrative in context.

Four facts are *derived* rather than stated, and substituting them independently
would make the passage inconsistent:
  Margaret's arrival  = resume time + offset minutes (meridiem included)
  counterfactual loss = hours lost + how much earlier the line severed
  'the third cup'     = cups brought + 1          (ordinal)
  'Three people'      = cups brought + 1          (cardinal)
So values are drawn first and the derived strings computed from them.

The clock is decorative where no question tests it: a randomised sever/resume
pair need not span exactly the stated hours lost, and nothing in the QA set asks
for that span. The counterfactual that *is* asked -- severed delta hours earlier
-- is computed from the stated loss, so it stays exact.

Substitution is two-phase -- originals to sentinels, sentinels to values -- so a
replacement can never be rewritten by a later rule.
"""
import re, random

NUM=("zero one two three four five six seven eight nine ten eleven twelve thirteen "
     "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()+[
     f"twenty-{w}" for w in "one two three four five six seven eight nine".split()]+["thirty"]
ORD="zeroth first second third fourth fifth sixth seventh eighth ninth tenth \
eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth \
eighteenth nineteenth twentieth".split()
cap=lambda s: s[0].upper()+s[1:]

POOL=dict(
 first1=["Elena","Priya","Ingrid","Rosa"],       last1=["Vargas","Okafor","Lindqvist","Duarte"],
 first2=["Marcus","Dmitri","Tomas","Kwame"],     last2=["Chen","Silva","Novak","Adeyemi"],
 first3=["Samir","Hassan","Viktor","Elias"],     last3=["Patel","Reyes","Kowalski","Mbeki"],
 first4=["Margaret","Beatrice","Yolanda","Helen"],last4=["Holt","Ferraro","Nakamura","Osei"],
 place=["Crescent Bay","Tern Point","Halloway Cove","Bishop Reach"],
 org=["Ocean Preservation Foundation","Coastal Waters Trust","Reef Legacy Institute",
      "Blue Horizon Foundation"],
 cafe=["Harbor Café","Dockside Roasters","Lantern Bakery","Pier Nine"],
 vehicle=["pickup truck","cargo van","station wagon","utility jeep"],
 color=["blue","green","silver","crimson"],      direction=["northeast","southwest","northwest","southeast"],
 animal_adj=["harbor","river","grey","brown"],   animal=["seal","otter","heron","pelican"],
 drink=["coffee","tea","cocoa","cider"],         style=["black","plain","unsweetened","strong"],
 additive=["oat milk","honey","cream","sugar"],  cleaner=["vinegar","alcohol","solvent","detergent"],
 illness=["flu","cold","fever","measles"],       month=["March","June","October","January"],
 day_visit=["Tuesday","Thursday","Friday","Sunday"],
 day_storm=["Monday","Wednesday","Saturday","Thursday"],
 age1=[34,41,29,52],  age2=[22,25,19,31],  wind=[40,55,25,65],  thresh=[48,72,36,24],
 money1=["$50,000","$80,000","$25,000","$120,000"], money2=["$2 million","$5 million","$800,000","$3 million"],
 years=[3,5,7,9],     n_boards=[2,3,4,5],  n_cables=[3,4,5,6],  n_cups=[2,3,4],
 n_minutes=[5,8,12,15], n_earlier=[30,45,20,60], hours_lost=[12,10,16,20], delta=[2,3,4,5],
 t_arrive=["7:45","6:15","8:20","5:40"], t_call=["6:30","5:10","4:55","7:05"],
 t_fixed=["9:30","8:45","10:15","11:20"], t_sever_h=[10,9,11,8], t_sever_m=[47,25,38,52],
 t_resume_h=[10,9,11,8], t_resume_m=[47,25,38,52], offset=[22,35,17,41],
)

def assign(rng, idx=None, nvar=None):
    """One consistent cast.

    idx selects cyclically instead of randomly, which balances how often each pool
    value is used -- but note the whole assignment then repeats with period
    lcm(pool sizes), so idx mode yields far fewer distinct casts than documents
    (12 here, not 24). That is what makes it the low-diversity arm."""
    a={}
    for k,v in POOL.items():
        off=sum(ord(c) for c in k)          # stable: hash() is salted per process
        a[k]= v[rng.randrange(len(v))] if idx is None else v[(idx+off)%len(v)]
    # ---- derived ----
    a["t_sever"]=f"{a['t_sever_h']}:{a['t_sever_m']:02d}"
    a["t_resume"]=f"{a['t_resume_h']}:{a['t_resume_m']:02d}"
    cf=a["t_sever_h"]-a["delta"]                                   # counterfactual sever time
    a["t_cf"]=f"{cf if cf>0 else cf+12}:{a['t_sever_m']:02d}"
    m=a["t_resume_m"]+a["offset"]; h=a["t_resume_h"]+m//60          # Margaret's arrival
    # the meridiem travels with the sum: 11:52 AM + 35 min is 12:27 PM, not AM
    a["t_margaret"]=f"{h if h<=12 else h-12}:{m%60:02d} " + ("PM" if h>=12 else "AM")
    a["hours_cf"]=cap(NUM[a["hours_lost"]+a["delta"]])              # 'Fourteen hours'
    a["cups_plus"]=NUM[a["n_cups"]+1]                               # 'three coffee cups'
    a["ord_cup"]=ORD[a["n_cups"]+1]                                 # 'the third cup'
    for k in ("n_boards","n_cables","n_cups","n_minutes","years"): a[f"w_{k}"]=NUM[a[k]]
    return a

# original literal -> slot, most specific first; a later rule can never touch an
# earlier rule's output because phase one writes sentinels, not values
SUBS=[
 (r"Ocean Preservation Foundation","org"), (r"Crescent Bay","place"), (r"Harbor Café","cafe"),
 (r"harbor seal","ANIMAL_FULL"), (r"pickup truck","vehicle"), (r"oat milk","additive"),
 (r"22 minutes","offset_min"), (r"30 minutes","n_earlier_m"), (r"12 hours","hours_lost_h"),
 (r"Fourteen hours","hours_cf_h"), (r"three years ago","years_ago"),
 (r"two replacement circuit boards","boards"), (r"three fiber-optic cables","cables"),
 (r"two large coffee cups","cups_large"), (r"two coffee cups","cups_plain"),
 (r"three coffee cups","cups_plus_c"), (r"Three people","people_c"),
 (r"five minutes","minutes_m"), (r"\bthird\b","ord_cup"),
 (r"7:45","t_arrive"), (r"6:30","t_call"), (r"9:30","t_fixed"),
 (r"8:47 PM","t_cf_pm"), (r"10:47 PM","t_sever_pm"), (r"10:47 AM","t_resume_am"),
 (r"11:09 AM","t_margaret"),
 (r"\b34\b","age1"), (r"\b22\b","age2"), (r"\b40\b","wind"), (r"\b48\b","thresh"),
 (r"\$50,000","money1"), (r"\$2 million","money2"),
 (r"\bVargas\b","last1"), (r"\bElena\b","first1"), (r"\bChen\b","last2"), (r"\bMarcus\b","first2"),
 (r"\bPatel\b","last3"), (r"\bSamir\b","first3"), (r"\bHolt\b","last4"), (r"\bMargaret\b","first4"),
 (r"\bblue\b","color"), (r"\bnortheast\b","direction"), (r"\bseal\b","animal"),
 (r"\bcoffee\b","drink"), (r"\bblack\b","style"), (r"\bvinegar\b","cleaner"),
 (r"\bflu\b","illness"), (r"\bMarch\b","month"), (r"\bTuesday\b","day_visit"), (r"\bMonday\b","day_storm"),
]

def values(a):
    """sentinel key -> replacement text"""
    v=dict(a)
    v.update({"ANIMAL_FULL":f"{a['animal_adj']} {a['animal']}",
        "offset_min":f"{a['offset']} minutes", "n_earlier_m":f"{a['n_earlier']} minutes",
        "hours_lost_h":f"{a['hours_lost']} hours", "hours_cf_h":f"{a['hours_cf']} hours",
        "years_ago":f"{a['w_years']} years ago",
        "boards":f"{a['w_n_boards']} replacement circuit boards",
        "cables":f"{a['w_n_cables']} fiber-optic cables",
        "cups_large":f"{a['w_n_cups']} large {a['drink']} cups",
        "cups_plain":f"{a['w_n_cups']} {a['drink']} cups",
        "cups_plus_c":f"{a['cups_plus']} {a['drink']} cups",
        "people_c":f"{cap(a['cups_plus'])} people",
        "minutes_m":f"{a['w_n_minutes']} minutes",
        "t_cf_pm":f"{a['t_cf']} PM", "t_sever_pm":f"{a['t_sever']} PM",
        "t_resume_am":f"{a['t_resume']} AM"})
    return {k:str(x) for k,x in v.items()}

def apply(text, a):
    for pat,key in SUBS: text=re.sub(pat, f"\x00{key}\x00", text)
    v=values(a)
    return re.sub(r"\x00(\w+)\x00", lambda m: v[m.group(1)], text)

ORIGINALS=["Elena","Marcus","Samir","Margaret","Vargas","Chen","Patel","Holt","Crescent Bay",
           "Ocean Preservation","Harbor Café","pickup truck","harbor seal","oat milk",
           "vinegar","flu","March","Tuesday","Monday","northeast","$50,000","$2 million",
           "7:45","6:30","9:30","10:47","8:47","11:09","Fourteen","third"]

def leftovers(text, a):
    """originals that survived -- empty unless a slot was missed. Values the new
    assignment happens to reuse are not leftovers, so they are excluded."""
    kept=set(str(x) for x in values(a).values())
    return [o for o in ORIGINALS if re.search(re.escape(o),text) and
            not any(o in k for k in kept)]

if __name__=="__main__":
    import sys
    rng=random.Random(int(sys.argv[1]) if len(sys.argv)>1 else 1)
    a=assign(rng); txt=open("marine.txt").read()
    out=apply(txt,a); lo=leftovers(out,a)
    print("leftover originals:", lo if lo else "none")
    for para in out.split("\n\n")[:1]+out.split("\n\n")[2:4]:
        print("\n"+para[:640])
