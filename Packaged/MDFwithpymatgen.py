from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.composition import Composition
from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.analysis.reaction_calculator import BalancedReaction
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram as PourbaixDiagramMP

from ase.data import chemical_symbols, atomic_numbers

# brute-force combinations, all possible combinations
from .my_pourbaix_brute import PourbaixDiagram as PourbaixDiagramBrute
# includes all possible all-ion and all-solid combinations
from .my_pourbaix_allpc import PourbaixDiagram as PourbaixDiagramAllPC
# three groups of combinations reduced: all, the solid entries, and the ion entries; default version 
from .pourbaix_final import PourbaixDiagram

import numpy as np
import matplotlib.pyplot as plt

# stable oxidation states in water
ox_states = {'H': [-1, 1],
 'He': [],
 'Li': [-1, 1],
 'Be': [2],
 'B': [3],
 'C': [0],
 'N': [],
 'O': [],
 'F': [],
 'Ne': [],
 'Na': [-1, 1],
 'Mg': [2],
 'Al': [3],
 'Si': [4],
 'P': [-0.5, 0],
 'S': [0],
 'Cl': [],
 'Ar': [],
 'K': [-1, 1],
 'Ca': [2, 4],
 'Sc': [3],
 'Ti': [2, 3, 4, 6],
 'V': [2, 3, 4, 5],
 'Cr': [2, 3, 4, 6],
 'Mn': [2, 3, 4, 6, 7],
 'Fe': [2, 3, 6],
 'Co': [2, 3, 4],
 'Ni': [2, 3, 4],
 'Cu': [1, 2],
 'Zn': [2],
 'Ga': [3],
 'Ge': [2, 4],
 'As': [3, 5],
 'Se': [0],
 'Br': [],
 'Kr': [],
 'Rb': [-1, 1],
 'Sr': [2, 4],
 'Y': [3],
 'Zr': [4],
 'Nb': [2, 4, 5],
 'Mo': [4, 6],
 'Tc': [4, 6, 7],
 'Ru': [3, 4, 8],
 'Rh': [1, 2, 3, 4],
 'Pd': [0, 2, 4, 6],
 'Ag': [1, 2, 3],
 'Cd': [2],
 'In': [3],
 'Sn': [2, 4],
 'Sb': [3, 4, 5],
 'Te': [0, 2, 4, 6],
 'I': [0],
 'Xe': [],
 'Cs': [-1, 1],
 'Ba': [2, 4],
 'La': [3],
 'Ce': [3, 4],
 'Pr': [3, 4],
 'Nd': [3],
 'Pm': [3],
 'Sm': [3],
 'Eu': [2, 3],
 'Gd': [3],
 'Tb': [3],
 'Dy': [3],
 'Ho': [3],
 'Er': [3],
 'Tm': [3],
 'Yb': [3],
 'Lu': [3],
 'Hf': [4],
 'Ta': [5],
 'W': [4, 5, 6],
 'Re': [3, 4, 6, 7],
 'Os': [4, 8],
 'Ir': [3, 4],
 'Pt': [2, 4, 6],
 'Au': [1, 3, 4],
 'Hg': [1, 2],
 'Tl': [1, 3],
 'Pb': [2, 3, 4],
 'Bi': [3, 4, 5],
 'Po': [4, 6],
 'At': [-1, 1],
 'Rn': [2],
 'Fr': [1],
 'Ra': [2],
 'Ac': [3],
 'Th': [4],
 'Pa': [5],
 'U': [2, 3, 4, 5, 6],
 'Np': [3, 4, 5, 6],
 'Pu': [3, 4, 5, 6],
 'Am': [3, 4, 5, 6],
 'Cm': [3],
 'Bk': [3],
 'Cf': [3],
 'Es': [3],
 'Fm': [3],
 'Md': [3],
 'No': [2],
 'Lr': [3],
 'Rf': [4],
 'Db': [5],
 'Sg': [6],
 'Bh': [7],
 'Hs': [8],
 'Mt': [],
 'Ds': [],
 'Rg': [],
 'Cn': [2],
 'Nh': [],
 'Fl': [],
 'Mc': [],
 'Lv': [],
 'Ts': [],
 'Og': []}

skip = ['He', 'N', 'O', 'F', 'Ne', 'Cl', 'Ar', 'Br', 'Kr', 'Xe', 'Mt', 'Ds', 'Rg', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
only_solid = ['C', 'S', 'Se', 'I']
no_ions_for = ['H','He','O','Ne','Ar','Kr','Xe','Pm','Po','At','Rn','Fr','Ra','Ac','Pa','Np','Am','Cm','Bk',
               'Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']
overall_skipping = list(set(no_ions_for + skip))

# good = true if oxidation states are valid, false if not
# didnotcheck = metals, alloys, etc. that don't have H or O - used later to print which solids overall were not checked
# prints the solid's name if not valid
def check_ox_state(entry):
    didnotcheck = []
    good = True
    for st in entry.name.split(" + "):
        if "(s)" in st:
            tempcomp = Composition(st.split("(s)")[0])
            elees = list(tempcomp.as_dict().keys())
            if "H" in elees or "O" in elees:
                guesses = tempcomp.oxi_state_guesses(oxi_states_override = ox_states, target_charge = 0)
                if not guesses:
                    print("OXIDATION STATE:", st, "is not valid")
                    good = False
                    break
            else:
                didnotcheck.append(st)
    return good, didnotcheck

# ll = phase_type field of the Pourbaix entry
# single = true if single-element system
# must_have_ox is either false or a Pourbaix multi-entry - optional flag for if only products/product combinations 
# containing O are considered; for example, NiO + Al is fine, Ni + Al not fine
# P = passivation product, C = corrosion, N = other participating product
def classify_combo(ll, single = False, must_have_ox = False):
    passes = True
    if must_have_ox is not False:
        if "O" not in must_have_ox.composition.as_dict().keys():
            passes = False  
    if single:
        if ll == "Ion":
            return "C"
        elif passes and ll == "Solid":
            return "P"
        else:
            return "N"
    summm = 0
    for c in ll:
        if c == "Ion":
            summm += 0
        else:
            summm += 1
    if passes and summm == len(ll):
        return "P"
    elif summm == 0:
        return "C"
    else:
        return "N"

# makes automatic pymatgen names more understandable
# checking for HO2 (oxyhydroxide) --> O(OH), (HO) --> (OH), and HO --> (OH)
# can also be customized to change other display names; this function is applied to names on the plots and the MDF result
def pretty_name(name):
    if "HO" not in name:
        return name
    if "HO2" in name:
        name = name.replace("HO2", "O(OH)")
    if "(HO)" in name:
        name = name.replace("(HO)", "(OH)")
    if "HO" in name:
        alls = name.split("HO")
        for i in range(1, len(alls)):
            if alls[i][0] == "(" or alls[i][0] == "[":
                alls[i] = "(OH)" + alls[i]
            else:
                alls[i] = "HO" + alls[i]
        name = ''.join(alls)
    return name

# series of methods to find the oxide + ion that correspond to where the MDF was found
def find_ox_ion_pair(oxgrids, iongrids, minox, minion, index):
    for i, grid in enumerate(oxgrids):
        if grid[index] == minox[index]:
            ox_ind = i
            break
    for j, grid in enumerate(iongrids):
        if grid[index] == minion[index]:
            ion_ind = j
            break
    return (i, j)

def find_ox_ion_pairs(oxgrids, iongrids, minox, minion, indices):
    pairs = []
    for i in range(len(indices)):
        pairs.append(find_ox_ion_pair(oxgrids, iongrids, minox, minion, indices[i]))
    return pairs

def multiple_inds(difference, amt):
    flattened_ind = np.argsort(difference, axis = None)[:amt]
    inds = []
    for i in range(amt):
        inds.append(np.unravel_index(flattened_ind[i], difference.shape))
    return inds

# generates intermediates before actual calculation of the MDF:
# entries = computed Pourbaix entries that have gone through PourbaixDiagram module 
# single = single-element system
# ox_req = oxygen flag, meaning at least one of the products in the combination must have oxygen
def driving_force_2D_helper(entries, pH_range, potential_range, single = False, ox_req = False):
    len_pH = len(np.arange((pH_range[0][0]), (pH_range[0][1]), pH_range[1]))
    len_poten = len(np.arange((potential_range[0][0]), (potential_range[0][1]), potential_range[1]))
    pH, potential = np.mgrid[(pH_range[0][0]):(pH_range[0][1]):pH_range[1], potential_range[0][0]:potential_range[0][1]:potential_range[1]]

    ion_grids = {}
    oxide_grids = {}
    other_grids = {}
    
    for entry in entries:
        name = entry.name
        if ox_req:
            clas = classify_combo(entry.phase_type, single, must_have_ox = entry)
        else:
            clas = classify_combo(entry.phase_type, single)
        if clas == "P":
            oxide_grids[name] = entry.normalized_energy_at_conditions(pH, potential)
        elif clas == "C":
            ion_grids[name] = entry.normalized_energy_at_conditions(pH, potential)
        elif clas == "N":
            other_grids[name] = entry.normalized_energy_at_conditions(pH, potential)
    
    oxgrids = np.array(list(oxide_grids.values()))
    iongrids = np.array(list(ion_grids.values()))
    oxides_inv = list(oxide_grids.keys())
    ions_inv = list(ion_grids.keys())
    if not oxides_inv:
        print("No solids left!")
        return
    if not ions_inv:
        print("No ions!")
        return
    min_ox = np.amin(oxgrids, axis = 0)
    min_ion = np.amin(iongrids, axis = 0)
    if np.shape(min_ox) != np.shape(oxgrids[0]) or np.shape(min_ox) != np.shape(min_ion):
        print("Shapes do not match.")
        return
    difference = min_ox - min_ion
    
    if other_grids:
        othergrids = np.array(list(other_grids.values()))
        min_others = np.amin(othergrids, axis = 0)
        masking = (min_ox > min_others)
        difference[masking] = np.nan
        if np.isnan(difference).all():
            print("There is no place in the given range where passivation is the most stable:")
            print("pH range:", pH_range, "potential range:", potential_range)
            return
        
    return [oxide_grids, ion_grids, other_grids, oxgrids, iongrids, oxides_inv, ions_inv, min_ox, min_ion, difference]

# metals = metals involved in the system, in a list like ["Ni", "Al"]
# plot = True means the driving force diagrams will be plotted; plot = False means no plots
# zoom = which of the plots to zoom into; this makes one of the three plots larger + adds lines for other participating products
def driving_force_2D(intermediates, metals, pH_range, potential_range, plot = True, zoom = 1):
    [oxide_grids, ion_grids, other_grids, oxgrids, iongrids, oxides_inv, ions_inv, min_ox, min_ion, difference] = intermediates
    max_ind = np.unravel_index(np.nanargmin(difference, axis=None), difference.shape)
    (ox_ind, ion_ind) = find_ox_ion_pair(oxgrids, iongrids, min_ox, min_ion, max_ind)
    max_pH = round(pH_range[0][0] + max_ind[0]*pH_range[1], 5)
    max_poten = round(potential_range[0][0] + max_ind[1]*potential_range[1], 5)
    
    if plot:
        # plot slices
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6))
        # ax1: at the max's potential, ax2: at the max's pH, ax3: at middle potential
        pHr = np.arange((pH_range[0][0]), (pH_range[0][1]), pH_range[1])
        Ur = np.arange((potential_range[0][0]), (potential_range[0][1]), potential_range[1])
        len_pH = len(pHr)
        len_poten = len(Ur)
        midpoint = len_poten // 2
        mid_poten = round(potential_range[0][0] + midpoint*potential_range[1], 5)

        if len(metals) == 1:
            single = True
        else:
            single = False
            fig2, ax4 = plt.subplots(1, 1, figsize=(6,10))
            if zoom == 1:
                xaxis = pHr
                yind1 = np.arange(0, len_pH)
                yind2 = max_ind[1]
                xlab = "pH"
            elif zoom == 2:
                xaxis = Ur
                yind1 = max_ind[0]
                yind2 = np.arange(0, len_poten)
                xlab = "U (V)"
            elif zoom == 3:
                xaxis = pHr
                yind1 = np.arange(0, len_pH)
                yind2 = midpoint
                xlab = "pH"
            else:
                print("Will zoom in on plot 1 since no valid zoom number was given.")
                xaxis = pHr
                yind1 = np.arange(0, len_pH)
                yind2 = max_ind[1]
                xlab = "pH"

        for oxi in oxide_grids.keys():
            oxi_name = pretty_name(oxi)
            ax1.plot(pHr, oxide_grids[oxi][:, max_ind[1]], label = oxi_name)
            ax2.plot(Ur, oxide_grids[oxi][max_ind[0], :], label = oxi_name)
            ax3.plot(pHr, oxide_grids[oxi][:, midpoint], label = oxi_name)
            if not single:
                ax4.plot(xaxis, oxide_grids[oxi][yind1, yind2], label = oxi_name)
        for io in ion_grids.keys():
            ion_name = pretty_name(io)
            ax1.plot(pHr, ion_grids[io][:, max_ind[1]], '--', label = ion_name)
            ax2.plot(Ur, ion_grids[io][max_ind[0], :], '--', label = ion_name)
            ax3.plot(pHr, ion_grids[io][:, midpoint], '--', label = ion_name)
            if not single:
                ax4.plot(xaxis, ion_grids[io][yind1, yind2], '--', label = ion_name)
        for other in other_grids.keys():
            other_name = pretty_name(other)
            if not single:
                ax4.plot(pHr, other_grids[other][yind1, yind2], ':', color = "black", label = other_name)
            #ax4.plot(pHr, other_grids[other][yind1, yind2], ':', label = other_name)
        ax1.set_title("U = " + str(max_poten) + "V")
        ax2.set_title("pH = " + str(max_pH))
        ax3.set_title("U = " + str(mid_poten) + "V")
        ax1.set_xlabel("pH")
        ax1.set_ylabel("Chemical Potential (eV)")
        ax2.set_xlabel("U (V)")
        ax3.set_xlabel("pH")
        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.95, 1))
        if not single:
            ax4.set_title(("Plot " + str(zoom) + " zoomed in with mixed products plotted"))
            ax4.set_ylabel("Chemical Potential (eV)")
            ax4.set_xlabel(xlab)
            ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
    return {"oxide": pretty_name(oxides_inv[ox_ind]), 
            "ion": pretty_name(ions_inv[ion_ind]), 
            "pH": max_pH,
            "potential": max_poten,
            "diff (eV)": difference[max_ind]}

# method to print the MDF dictionary of oxide, ion, pH, potential, and max driving force value nicely
def print_dr_dict(max_dr):
    print("Passivation Products:", max_dr["oxide"])
    print("Corrosion Products:", max_dr["ion"])
    print("pH:", max_dr["pH"])
    print("Potential:", max_dr["potential"])
    print("Max Driving Force (eV):", max_dr["diff (eV)"])
    if max_dr["diff (eV)"] > 0:
        print("Max driving force displayed is positive: there is no place in the given range where passivation is the most stable.")
    plt.show()

# checks "metals" are all valid (more than just elemental solid possible in water, ions available from MP)
# only_solid_fine means it allows metals that have only 0 in their possible oxidation states in water
def check_metals(metals, only_solid_fine = False):
    if not only_solid_fine:
        for metal in metals:
            if metal in (overall_skipping + only_solid):
                print("This metal cannot be included:", metal)
                return False
    else:
        for metal in metals:
            if metal in (overall_skipping):
                print("This metal cannot be included:", metal)
                return False
    return True

# our filter: checks for oxidation state, hydrides, + only takes one entry per composition (the most stable)
import itertools
def my_filter(entries):
    solid_entries = [entry for entry in entries if entry.phase_type == "Solid"]
    sorted_entries = sorted(
        solid_entries,
        key=lambda x: (x.composition.reduced_composition, x.entry.energy_per_atom),
    )
    grouped_by_composition = itertools.groupby(sorted_entries, key=lambda x: x.composition.reduced_composition)
    min_entries = [list(grouped_entries)[0] for comp, grouped_entries in grouped_by_composition]
    all_entries = [entry for entry in entries if entry.phase_type == "Ion"]
    uncheckeds = []
    for en in min_entries:
        keep_going, unchecked = check_ox_state(en)
        if keep_going:
            eles = en.composition.as_dict().keys()
            if "H" in eles and "O" not in eles:
                continue
            all_entries.append(en)
        uncheckeds += unchecked
    print("Did not check the following:", list(set(uncheckeds)))
    return all_entries

# shortcut to get computed Pourbaix entries with our filter
def get_entries_with_my_filter(mpr, class_of_pourbaix, metals, metal_ratios, con_dict):
    if check_metals(metals):
        entries = mpr.get_pourbaix_entries(metals)
        filtered_ents = my_filter(entries)
        pbx = class_of_pourbaix(filtered_ents, comp_dict=metal_ratios,
                              conc_dict=con_dict, filter_solids=False)
        return pbx.all_entries
    return None

def construct_s_from_metals(mpr, metal, concen, myfilter = False):
    entries = mpr.get_pourbaix_entries(metal)
    if myfilter:
        pbx = PourbaixDiagram(my_filter(entries), conc_dict={metal: concen}, filter_solids = False)
    else:
        pbx = PourbaixDiagram(entries, conc_dict={metal: concen}, filter_solids = True)
    USING = pbx.all_entries
    return USING

# easy method for making a graph for single-element systems across the periodic table
# start and end are elemental numbers (ex: 21 - 30 means Sc to Zn, inclusive)
# ox_required = oxygen flag
# myfilter = if using our filter, = True, if using MP's filter, = True
def periodic_trend(mpr, start, end, pHrange, potentialrange, precision, concentration, ox_required = False, myfilter = False, only_solid_fine = False):
    if only_solid_fine:
        mask = overall_skipping
    else:
        mask = overall_skipping + only_solid
    want_metal_nums = [start, end]
    chem_symbols = np.array(chemical_symbols)
    want_metals = chem_symbols[np.arange(want_metal_nums[0], want_metal_nums[1] + 1, 1)]

    RANGE_PH = ([pHrange[0], pHrange[1]], precision)
    RANGE_POTENTIAL = ([potentialrange[0], potentialrange[1]], precision)

    CONCENTRATION = concentration

    drs = []
    plotted_metals = []
    oxs = []

    for metal in want_metals:
        if check_metals([metal]):
            entries = construct_s_from_metals(mpr, metal, CONCENTRATION, myfilter = myfilter)
            intermediates = driving_force_2D_helper(entries, RANGE_PH, RANGE_POTENTIAL, single = True, ox_req = ox_required)
            if not intermediates:
                print(metal, "was skipped due to no solids left.", "\n")
                continue
            max_dr = driving_force_2D(intermediates, [metal], RANGE_PH, RANGE_POTENTIAL)
            plotted_metals.append(metal)
            drs.append(max_dr["diff (eV)"])
            oxs.append(max_dr["oxide"])
            print_dr_dict(max_dr)
        else:
            print(metal, "was skipped.", "\n")
    
    xax = range(len(plotted_metals))
    plt.figure(figsize = (10, 6))
    plt.scatter(xax, drs)
    # add labels
    for i in range(len(plotted_metals)):
        plt.annotate(oxs[i], # this is the text
                     (xax[i],drs[i]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center',
                     backgroundcolor="w") # horizontal alignment can be left, right or center
    
    plt.xticks(xax, plotted_metals, size='small')
    plt.grid()
    plt.ylabel("Max Driving Force (eV)")
    # only negative MDFs are meaningful
    plt.ylim(top = 0.05)
    plt.show()
