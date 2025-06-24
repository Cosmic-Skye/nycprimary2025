#!/bin/bash

# Script to update model results and deploy with cache busting

echo "Running model..."
python model.py

echo "Adding cache-busting timestamp to HTML..."
# Update the dateModified in structured data to current date
current_date=$(date +%Y-%m-%d)
sed -i '' "s/\"dateModified\": \"[0-9-]*\"/\"dateModified\": \"$current_date\"/" index.html

# Note: For immediate cache invalidation of images on social media:
# - Facebook: Use their Sharing Debugger at https://developers.facebook.com/tools/debug/
# - Twitter: Cards automatically update, but may take a few minutes
# - LinkedIn: Use their Post Inspector at https://www.linkedin.com/post-inspector/