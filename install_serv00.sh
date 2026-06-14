#!/bin/bash
# Qythera installer for serv00 FreeBSD hosting

echo "=== Qythera serv00 Installer ==="

# Find web directory
WEB_DIR=""
for dir in ~/public_html ~/domains/*/public_html ~/www ~/html; do
    if [ -d "$dir" ] || [ -d "$(eval echo $dir)" ]; then
        WEB_DIR="$(eval echo $dir)"
        break
    fi
done

# Create public_html if it doesn't exist
if [ -z "$WEB_DIR" ]; then
    mkdir -p ~/public_html
    WEB_DIR=~/public_html
    echo "Created: $WEB_DIR"
fi

echo "Web directory: $WEB_DIR"

# Clone to temp location
TEMP=/tmp/qythera_install_$$
rm -rf "$TEMP"
git clone https://github.com/cpner/qythera.git "$TEMP"

# Copy PHP files to web directory
cp "$TEMP/php/api.php" "$WEB_DIR/"
cp "$TEMP/php/index.html" "$WEB_DIR/"

# Clean up
rm -rf "$TEMP"

echo ""
echo "Done! Open in browser:"
echo "  https://$(whoami).serv00.net/index.html"
echo ""
echo "Or test locally:"
echo "  cd $WEB_DIR && php -S localhost:8080"
