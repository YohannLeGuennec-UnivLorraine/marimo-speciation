# marimo Speciation Site

Ce repository exporte des notebooks marimo en HTML/WebAssembly et publie le resultat sur GitHub Pages.

## Ce qui est fait dans les notebooks

Tous les notebooks suivent la meme logique numerique:

1. Definition d'un systeme d'equilibre chimique (bilans + lois d'action de masse).
2. Resolution non lineaire avec `scipy.optimize.least_squares` en variables logarithmiques.
3. Verification des residus, des bilans de matiere et de l'electroneutralite (controle a posteriori).
4. Visualisations: pH, repartition/speciation et balayages parametriques.

### `notebooks/exemple_1_H2SO4_H2O.py`

- Systeme acido-basique `H2SO4/H2O`.
- Calcule l'equilibre pour une charge initiale donnee.
- Realise un balayage iteratif en `n(H2SO4)^0` et trace:
  - pH vs `n(H2SO4)^0`
  - concentrations d'equilibre (speciation) vs `n(H2SO4)^0`
  - concentrations vs pH
  - fractions de distribution sulfate.

### `notebooks/exemple_2_NiCl2_NH3_H2O.py`

- Systeme de complexation `NiCl2/NH3/H2O`.
- Prend en compte:
  - autoprotolyse de l'eau
  - couple `NH4+/NH3`
  - complexe `Ni(NH3)6^2+`.
- Fournit un etat d'equilibre (table des especes + diagnostics) puis un balayage en `n(NH3)^0`.
- Trace pH et speciation en fonction de `n(NH3)^0` et du pH.

### `notebooks/exemple_3_FeCl2_H2O.py`

- Systeme redox `FeCl2/H2O/O2`.
- Modele avec especes redox (`Fe3+`, `Fe2+`, `e-`) et `O2` dissous.
- Conversion des potentiels standards `E0` en constantes d'equilibre, puis resolution du systeme complet.
- Sorties principales:
  - pH d'equilibre
  - potentiel redox (Nernst)
  - speciation fer/oxygene
  - balayage en `n(FeCl2)^0`.

### `notebooks/exemple_4_AgCl_H2O.py`

- Systeme avec dissolution/precipitation `AgCl(s) <-> Ag+ + Cl-`.
- Resolution d'equilibre pour une charge initiale, puis balayage en `n(AgCl)^0`.
- Visualise l'effet sur le pH et sur les concentrations des especes dissoutes/solide.

## Structure `notebooks/` vs `apps/`

- `notebooks/`: export en mode edition (`--mode edit`), avec code visible/modifiable.
- `apps/`: export en mode execution (`--mode run --no-show-code`), interface simplifiee pour usage "application".

## Build

```bash
uv run .github/scripts/build.py
```

Le build genere le site dans `_site/`.
Par defaut, il synchronise aussi le resultat vers `docs/` (source GitHub Pages sans Actions).

Pour le servir en local:

```bash
python -m http.server -d _site
```

Puis ouvrir `http://localhost:8000`.

## Publication GitHub Pages

GitHub Pages ne permet pas d'utiliser `/_site` directement en mode "Deploy from a branch".
Les seules options sont `/(root)` ou `/docs`.

Configurer une seule fois dans GitHub:

1. `Settings` -> `Pages`
2. `Build and deployment` -> `Source`: **Deploy from a branch**
3. `Branch`: `main`
4. `Folder`: `/docs`

Ensuite:

1. Lancer le build localement (`uv run .github/scripts/build.py`)
2. Commit/push les changements (dont `docs/`)

GitHub Pages publiera le contenu de `docs/` automatiquement.

## Templates

Le template par defaut est `templates/tailwind.html.j2`.

Pour en utiliser un autre:

```bash
uv run .github/scripts/build.py --template templates/index.html.j2
```

Voir aussi `templates/README.md`.
