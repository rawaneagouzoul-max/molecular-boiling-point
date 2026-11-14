from functools import lru_cache

from flask import Flask, render_template, request, jsonify

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from chemicals import search_chemical, Tb as tb_experimentale
from thermo.group_contribution.joback import Joback

app = Flask(__name__)


# ----------------------------------------------------------------------
# Outils : valeur expérimentale de référence
# ----------------------------------------------------------------------

def smiles_canonique(texte):
    """Retourne le SMILES canonique RDKit (ou None si invalide)."""
    mol = Chem.MolFromSmiles(texte)
    return Chem.MolToSmiles(mol) if mol is not None else None


@lru_cache(maxsize=512)
def chercher_tb_experimentale(smiles_canon):
    """
    Cherche la température d'ébullition expérimentale (à 1 atm) dans la
    base de données de la bibliothèque `chemicals`.

    Piège évité : chercher "O" ou "C" directement renvoie l'oxygène
    atomique ou le carbone (éléments), pas l'eau ni le méthane.
    On essaie donc plusieurs formes de requête (InChI en premier) et on
    n'accepte un résultat que si sa structure est IDENTIQUE à celle
    demandée.
    """
    mol = Chem.MolFromSmiles(smiles_canon)
    if mol is None:
        return None

    requetes = [Chem.MolToInchi(mol), smiles_canon]

    # Forme de Kékulé (certaines bases n'acceptent pas les SMILES aromatiques)
    try:
        kekule = Chem.Mol(mol)
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        requetes.append(Chem.MolToSmiles(kekule, kekuleSmiles=True))
    except Exception:
        pass

    for requete in requetes:
        if not requete:
            continue

        try:
            meta = search_chemical(requete)
        except Exception:
            continue

        # Vérification : est-ce bien la même molécule ?
        if not meta.smiles or smiles_canonique(meta.smiles) != smiles_canon:
            continue

        tb = tb_experimentale(meta.CAS)

        if tb is not None:
            return {
                "Tb_K": float(tb),
                "nom": meta.common_name,
                "CAS": meta.CASs
            }

    return None


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.route("/")
def accueil():
    return render_template("index.html")


@app.route("/analyser", methods=["POST"])
def analyser():

    try:
        data = request.get_json()
        smiles = data.get("smiles", "").strip()

        # 1. Lire la structure
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return jsonify({
                "error": "Structure SMILES invalide."
            }), 400

        # 2. Informations de la molécule
        formule = rdMolDescriptors.CalcMolFormula(mol)
        masse = Descriptors.MolWt(mol)
        nombre_atomes_lourds = mol.GetNumAtoms()

        # 3. Ajouter les hydrogènes AVANT le comptage
        mol3d = Chem.AddHs(mol)

        nombre_total_atomes = mol3d.GetNumAtoms()
        nombre_liaisons = mol3d.GetNumBonds()

        # 4. Générer la géométrie 3D
        parametres = AllChem.ETKDGv3()
        parametres.randomSeed = 42

        statut_3d = AllChem.EmbedMolecule(mol3d, parametres)

        if statut_3d != 0:
            return jsonify({
                "error": "Impossible de générer la structure 3D."
            }), 422

        # Optimisation géométrique
        AllChem.MMFFOptimizeMolecule(mol3d)

        # 5. Décrire tous les atomes
        atom_details = []

        for atom in mol3d.GetAtoms():
            atom_details.append({
                "index": atom.GetIdx(),
                "symbol": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "charge": atom.GetFormalCharge()
            })

        # 6. Décrire toutes les liaisons
        bond_details = []

        for bond in mol3d.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            bond_details.append({
                "atom1": atom1.GetIdx(),
                "atom2": atom2.GetIdx(),
                "symbol1": atom1.GetSymbol(),
                "symbol2": atom2.GetSymbol(),
                "type": str(bond.GetBondType())
            })

        # 7. Préparer les données 3D
        molblock = Chem.MolToMolBlock(mol3d)

        # 8. Estimation par la méthode de Joback
        Tb = None
        Tb_C = None
        groupes = {}
        message_joback = None

        try:
            joback = Joback(smiles)

            if joback.status == "OK":
                estimation = joback.estimate(callables=False)
                Tb = estimation.get("Tb")

                if Tb is not None:
                    Tb_C = Tb - 273.15

                groupes = {
                    str(k): int(v)
                    for k, v in joback.counts.items()
                }

            else:
                message_joback = (
                    "La méthode de Joback ne s'applique pas à cette "
                    "molécule : certains atomes n'appartiennent à aucun "
                    "groupe de sa table (c'est le cas de l'eau, du méthane "
                    "et de beaucoup de très petites molécules)."
                )

        except Exception as erreur_joback:
            message_joback = str(erreur_joback)

        # 9. Valeur expérimentale de référence (si connue)
        reference = chercher_tb_experimentale(Chem.MolToSmiles(mol))

        Tb_exp = None
        Tb_exp_C = None
        nom_substance = None
        cas = None
        ecart_K = None
        ecart_pct = None

        if reference is not None:
            Tb_exp = reference["Tb_K"]
            Tb_exp_C = Tb_exp - 273.15
            nom_substance = reference["nom"]
            cas = reference["CAS"]

            if Tb is not None:
                ecart_K = Tb - Tb_exp
                ecart_pct = 100 * ecart_K / Tb_exp

        # 10. Envoyer tous les résultats au site
        return jsonify({

            "smiles": smiles,
            "formula": formule,
            "mass": round(masse, 3),

            "atoms": nombre_total_atomes,
            "heavy_atoms": nombre_atomes_lourds,
            "bonds": nombre_liaisons,

            # Estimation Joback
            "Tb_K": round(Tb, 2) if Tb is not None else None,
            "Tb_C": round(Tb_C, 2) if Tb_C is not None else None,
            "calculation_available": Tb is not None,
            "joback_message": message_joback,
            "groups": groupes,

            # Valeur expérimentale de référence
            "Tb_exp_K": round(Tb_exp, 2) if Tb_exp is not None else None,
            "Tb_exp_C": round(Tb_exp_C, 2) if Tb_exp_C is not None else None,
            "nom_substance": nom_substance,
            "cas": cas,
            "ecart_K": round(ecart_K, 2) if ecart_K is not None else None,
            "ecart_pct": round(ecart_pct, 2) if ecart_pct is not None else None,

            "atom_details": atom_details,
            "bond_details": bond_details,

            "molblock": molblock

        })

    except Exception as erreur:

        return jsonify({
            "error": str(erreur)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)