# BRANCH OFICIAL ENTREGA 3: `Branch_final_con_sensibilidades `

# 🧭 Guía rápida de trabajo en este repositorio

Objetivo: que todos podamos trabajar ordenados en ramas, usar consola para los commits, pull, push, y tener el entorno configurado igual.

---

## 🌱 1. Crear y entrar a una branch

> **Regla:** nunca trabajar directo en `main`.

```bash
#partir desde main actualizado
git checkout main
git pull origin main
```
crear y cambiarte a tu branch
```bash
git checkout -b feat/tu-feature
# ejemplo: git checkout -b feat/modelo-xgboost
```

Para volver a una branch existente:

```bash
git checkout nombre-de-branch
```
Listar todas las branches:

```bash
git branch -a
```
## ⚙️ 2. Crear el entorno del proyecto
No subimos `.venv` ni `.env`.
Cada persona crea su propio entorno local.

### 🔹 Opción A: pip + venv
```bash
# crear entorno virtual
python -m venv .venv

# activar entorno
# mac / linux
source .venv/bin/activate
# windows (powershell)
.venv\Scripts\Activate.ps1

# instalar dependencias
pip install -r requirements.txt
```

Para congelar dependencias nuevas:

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "chore: actualiza dependencias"
git push
```


## 🔁 3. Ciclo básico de trabajo con Git
```bash
# ver cambios
git status

# agregar archivos
git add .
# o específico: git add src/modelo.py

# hacer commit
git commit -m "feat: agrega modelo xgboost inicial"

# primer push de tu branch
git push -u origin nombre-de-branch

# siguientes veces
git push
```

Traer cambios de `main` y rebasear tu branch:

```bash
git checkout main
git pull origin main

git checkout nombre-de-branch
git rebase main

# si hay conflictos, resolverlos y continuar
git add .
git rebase --continue
git push --force-with-lease
```

## 🔀 4. Pull Request (PR)
1. Subir tu branch
2. En GitHub → Compare & Pull Request
3. Escribir una descripción breve
4. Pedir revisión a otro del grupo
5. Usar Squash and Merge
6. Borrar la branch cuando se apruebe

## 🧩 5. Comandos útiles
```bash
git status
git log --oneline --graph --decorate --all
git branch -a
git checkout -
git fetch origin
git stash
git stash pop
```

## 🔒 6. Reglas de data, modelos y secretos
- No subir `.venv`, `env/`ni `.env`
- No subir claves, contraseñas ni gurobi.lic
- Si los datasets o modelos pesan poco, se pueden subir
- Si pesan más de 100 MB, usar Git LFS


## 📂 7. Estructura recomendada del proyecto
```bash
.
├─ src/              # código fuente
├─ notebooks/        # jupyter notebooks
├─ tests/            # pruebas unitarias
├─ docs/             # documentación
├─ data/             # datasets (si se versionan)
├─ models/           # modelos entrenados (si se versionan)
├─ .env.example      # ejemplo de variables locales
├─ .gitignore
└─ README.md
```

## 💡 Recomendaciones finales
- Antes de comenzar a trabajar, hacer siempre git pull origin main
- Crear una branch nueva para cada cosa importante
- Hacer commits chicos y con mensajes claros
- Revisar bien antes de hacer merge a main

Si algo falla, no forzar nada sin entenderlo primero 😅




