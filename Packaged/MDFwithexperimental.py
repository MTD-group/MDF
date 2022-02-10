from ase.data import chemical_symbols, atomic_numbers

from pymatgen.core.composition import Composition
from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.analysis.reaction_calculator import BalancedReaction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


from .experimental_data import *
# constants
# units: eV/V/atom
FARADAYS = 1
# units: K
TEMP = 25 + 273
# units: eV/K/atom
R = 8.63 * (10**-5)
RT = TEMP * R
# units: eV
H2O_gibbs = -2.4577835 
LN_10 = np.log(10)

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

ox_names_by_metal = {}
for ox in all_oxides.keys():
    if len(all_oxides[ox]["inv_elements"]) == 2:
        ele = all_oxides[ox]["inv_elements"][1]
        if ox_names_by_metal.get(ele):
            ox_names_by_metal[ele].append(ox)
        else:
            ox_names_by_metal[ele] = [ox]

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

# explained in pymatgen MDF
def check_ox_state(entry_name):
    didnotcheck = []
    good = True
    tempcomp = Composition(entry_name)
    elees = list(tempcomp.as_dict().keys())
    if "H" in elees or "O" in elees:
        guesses = tempcomp.oxi_state_guesses(oxi_states_override = ox_states, target_charge = 0)
        if not guesses:
            print("OXIDATION STATE:", st, "is not valid")
            good = False
    else:
        didnotcheck.append(st)
    return good, didnotcheck

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

# check oxide coeff is 1 - returns None if not
# [A, B, C, D]: A = num elections, B = num h plus, C = list of metal counts, D = num water
def rxn_coeffs(spec, charge, elements):
    if 'OH' in spec:
        spec = spec.replace("OH", "HO")
    water = Composition("H2O")
    reactants = [water]
    metals = []
    for ele in elements:
        if ele == 'O':
            continue
        reactants.append(Composition(ele))
        metals.append(ele)
    ox = Composition(spec)
    h2 = Composition("H2")
    rxn = BalancedReaction.from_string(str(Reaction(reactants, [ox, h2])))
    if len(rxn.all_comp) == 0:
        return [charge, 0, metals, [1.0], 0]
    if rxn.get_coeff(ox) != 1:
        return None
    metal_coeffs = []
    for i in range(1, len(reactants)): 
         metal_coeffs.append(np.abs(rxn.get_coeff(reactants[i])))
    h_plus_coeff = 2*rxn.get_coeff(h2)
    water_coeff = np.abs(rxn.get_coeff(water)) 
    e_coeff = charge + h_plus_coeff
    return [e_coeff, h_plus_coeff, metals, metal_coeffs, water_coeff]

def driving_force_2D_helper(species, metal, pH_range, potential_range, concen, single = False):
    len_pH = len(np.arange((pH_range[0][0]), (pH_range[0][1]), pH_range[1]))
    len_poten = len(np.arange((potential_range[0][0]), (potential_range[0][1]), potential_range[1]))
    pH, potential = np.mgrid[(pH_range[0][0]):(pH_range[0][1]):pH_range[1], potential_range[0][0]:potential_range[0][1]:potential_range[1]]
    
    # chemical potential equation helper method
    def chem_potential_2D(coeffs, ion, specs_gibbs):
        A = coeffs[0]
        B = coeffs[1]
        C = np.sum(coeffs[3])
        D = coeffs[4]
        subtracted_section = A*FARADAYS*potential + B*RT*LN_10*pH + D*H2O_gibbs
        if ion:
            return (specs_gibbs + RT*np.log(concen) - subtracted_section) / C
        else:
            return (specs_gibbs - subtracted_section) / C

    ion_grids = {}
    oxide_grids = {}
    uncheckeds = []
    
    oxide_grids[metal] = np.zeros((len_pH, len_poten))
    
    for spec in species.keys():
        if species[spec][0] == "ion":
            is_ion = True
            gibbs = expanded_all_ions[spec]["energy"]
            spec_without_ion = expanded_all_ions[spec]["without_charge"]
            charge = expanded_all_ions[spec]["charge"]
        else:
            keep_going, unchecked = check_ox_state(spec)
            uncheckeds += unchecked
            if not keep_going:
                continue
            is_ion = False
            gibbs = all_oxides[spec]['formation_energy_per_fu']
            spec_without_ion = spec
            charge = 0
        coeffs = rxn_coeffs(spec_without_ion, charge, species[spec][1])
        if is_ion:
            ion_grids[spec] = chem_potential_2D(coeffs, is_ion, gibbs)
        else:
            oxide_grids[spec] = chem_potential_2D(coeffs, is_ion, gibbs)

    if len(uncheckeds) > 0:
        print("Did not check the following:", list(set(uncheckeds)))
    
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
    min_ox_ind = np.argmin(oxgrids, axis = 0)
    min_ox = np.amin(oxgrids, axis = 0)
    min_ion = np.amin(iongrids, axis = 0)
    if np.shape(min_ox) != np.shape(oxgrids[0]) or np.shape(min_ox) != np.shape(min_ion) or np.shape(min_ox) != np.shape(min_ox_ind):
        print("Shapes do not match.")
        return
    difference = min_ox - min_ion
        
    return [oxide_grids, ion_grids, oxgrids, iongrids, oxides_inv, ions_inv, min_ox_ind, min_ox, min_ion, difference]

def driving_force_2D(intermediates, metals, pH_range, potential_range, plot = True, zoom = 1):
    [oxide_grids, ion_grids, oxgrids, iongrids, oxides_inv, ions_inv, min_ox_ind, min_ox, min_ion, difference] = intermediates
    
    ox_to_mdf = {}
    for i, oxi in enumerate(oxide_grids.keys()):
        temp_diff = deepcopy(difference)
        temp_diff[min_ox_ind != i] = np.nan
        if np.isnan(temp_diff).all():
            print("There is no place in the given range where", oxi,  "is the most stable:")
            continue
        max_ind = np.unravel_index(np.nanargmin(temp_diff, axis=None), temp_diff.shape)
        (ox_ind, ion_ind) = find_ox_ion_pair(oxgrids, iongrids, min_ox, min_ion, max_ind)
        max_pH = round(pH_range[0][0] + max_ind[0]*pH_range[1], 5)
        max_poten = round(potential_range[0][0] + max_ind[1]*potential_range[1], 5)
        ox_to_mdf[oxi] = {"oxide": pretty_name(oxides_inv[ox_ind]),
                          "ion": pretty_name(ions_inv[ion_ind]),
                          "pH": max_pH,
                          "potential": max_poten,
                          "diff (eV)": difference[max_ind]}
    
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
    return ox_to_mdf

# prints dictionary of MDF dictionaries
def print_dr_dict(max_drs):
    for max_dr in max_drs.values():
        print()
        print("Passivation Products:", max_dr["oxide"])
        print("Corrosion Products:", max_dr["ion"])
        print("pH:", max_dr["pH"])
        print("Potential:", max_dr["potential"])
        print("Max Driving Force (eV):", max_dr["diff (eV)"])
        if max_dr["diff (eV)"] > 0:
            print("Max driving force displayed is positive: there is no place in the given range where passivation is the most stable.")
    plt.show()

# search for corresponding ions - just ignore when get 'O'
# then error if any don't have ions
def get_corresponding_ions(oxide_p_formula):
    end_dict = {}
    stripped_formula = oxide_p_formula.strip()
    if all_oxides.get(stripped_formula) is None:
        print("Could not find this oxide:", stripped_formula)
        return
    for i in all_oxides[stripped_formula]['inv_elements']:
        if i == 'O':
            continue
        if all_ions.get(i) is None:
            return None
        for ion in all_ions[i]:
            if all_ions[i][ion]["in_consideration"]:
                end_dict[ion] = ["ion", [i]]
    return end_dict

# get all species based on the oxide list
def construct_s_from_oxides(oxides, hide = []):
    spe = {}
    ionlist = {}
    for spec in oxides:
        if all_oxides.get(spec):
            spe[spec] = ["oxide", all_oxides[spec]['inv_elements']]
        ions = get_corresponding_ions(spec)
        if ions is None:
            print("Do not have ions for this oxide:", spec)
            return None
        ionlist.update(ions)
    spe.update(ionlist)
    if len(hide) > 0:
        for hideit in hide:
            temp_capture = spe.pop(hideit, None)
            if temp_capture is None:
                print("Species construction - Hide:", hideit, "was not found.")
    return spe


# sorting the dictionary of MDF dictionaries
def sorted_top(max_dr, top = 2):
    drs = []
    oxs = np.array(list(max_dr.keys()))
    for key in oxs:
        drs.append(max_dr[key]["diff (eV)"])
    ind = np.argsort(drs)
    drs = np.array(drs)
    if len(ind) < top:
        return oxs[ind], drs[ind]
    else:
        return oxs[ind[:top]], drs[ind[:top]]


# plots the best and second best passivation product of each element (if they exist)
def periodic_trend(start, end, pHrange, potentialrange, precision, concentration):
    want_metal_nums = [start, end]
    chem_symbols = np.array(chemical_symbols)
    want_metals = chem_symbols[np.arange(want_metal_nums[0], want_metal_nums[1] + 1, 1)]
    
    RANGE_PH = ([pHrange[0], pHrange[1]], precision)
    RANGE_POTENTIAL = ([potentialrange[0], potentialrange[1]], precision)

    CONCENTRATION = concentration
    plotted_metals = []

    bestdrs = []
    bestoxs = []
    
    bestdrs2 = []
    bestoxs2 = []
    
    all_mdf_dicts = []

    for metal in want_metals:
        if check_metals([metal]):
            SPECIES = construct_s_from_oxides(ox_names_by_metal[metal])
            intermediates = driving_force_2D_helper(SPECIES, metal, RANGE_PH, RANGE_POTENTIAL, CONCENTRATION)
            if not intermediates:
                print(metal, "was skipped due to no solids left.", "\n")
                continue
            max_dr = driving_force_2D(intermediates, [metal], RANGE_PH, RANGE_POTENTIAL)
            all_mdf_dicts.append(max_dr)
            top_oxs, top_vals = sorted_top(max_dr, top = 2)
            plotted_metals.append(metal)
            bestdrs.append(top_vals[0])
            bestoxs.append(top_oxs[0])
            if len(top_oxs) > 1:
                bestdrs2.append(top_vals[1])
                bestoxs2.append(top_oxs[1])
            else:
                bestdrs2.append(1)
                bestoxs2.append(1)
            print_dr_dict(max_dr)
        else:
            print(metal, "was skipped.", "\n")
    
    plt.figure(figsize = (16, 6))

    xax = range(len(plotted_metals))
    plt.scatter(xax, bestdrs, label = "best")
    if bestdrs2:
        plt.scatter(xax, bestdrs2, label = "second best", marker = "*")

    # add labels
    for i in range(len(plotted_metals)):
        plt.annotate(bestoxs[i], # this is the text
                     (xax[i],bestdrs[i]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-20), # distance from text to points (x,y)
                     ha='center',
                     backgroundcolor = "w") # horizontal alignment can be left, right or center
        if bestoxs2:
            plt.annotate(bestoxs2[i], # this is the text
                         (xax[i],bestdrs2[i]), # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center',
                         backgroundcolor = "w") # horizontal alignment can be left, right or center
    
    plt.xticks(xax, plotted_metals, size='small')
    plt.grid()
    plt.ylabel("Max Driving Force (eV)")
    plt.ylim(top = 0.05)
    plt.legend(loc='lower left')
    plt.show()
    return plotted_metals, bestoxs, bestdrs, bestoxs2, bestdrs2, all_mdf_dicts



