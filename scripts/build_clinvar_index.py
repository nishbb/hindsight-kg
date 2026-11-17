"""Build the local dated ClinVar SQLite index used by the dossier (about 3 min, one-off)."""

from kggc.variant_dossier.clinvar_local import build_index

if __name__ == "__main__":
    print(build_index())
