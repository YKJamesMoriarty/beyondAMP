clone_repo() {
    local repo_url=$1
    local target_dir=$2

    if [ ! -d "$target_dir/.git" ]; then
        echo "→ Cloning $repo_url to $target_dir ..."
        mkdir -p "$(dirname "$target_dir")"
        git clone "$repo_url" "$target_dir"
    else
        echo "✔ Repository already exists at $target_dir, skipping clone."
    fi
}

update_repo() {
    local repo_url=$1
    local target_dir=$2

    echo "🔄 Updating repository at $target_dir ..."

    if [ ! -d "$target_dir/.git" ]; then
        echo "→ No git repo found, recloning..."
        rm -rf "$target_dir"
        mkdir -p "$(dirname "$target_dir")"
        git clone "$repo_url" "$target_dir"
        return
    fi

    echo "→ Found existing repo, force updating ..."
    (
        cd "$target_dir" || exit 1
        git fetch --all
        git reset --hard origin/HEAD
        git clean -fd
        echo "✔ Updated $repo_url"
    )
}


install_modules() {
    local modules=("$@")

    echo "📦 Installing local editable modules..."
    for module in "${modules[@]}"; do
        if [ -d "$module" ]; then
            echo "→ Installing $module ..."
            pip install -e "$module"
        else
            echo "⚠ Skipped $module (directory not found)"
        fi
    done
}


update_repo https://github.com/Renforce-Dynamics/assetslib.git ./data/assets/assetslib
update_repo https://github.com/Renforce-Dynamics/robotlib.git ./source/robotlib
