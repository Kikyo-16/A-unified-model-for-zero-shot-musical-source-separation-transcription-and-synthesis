git rev-list --objects --all | grep demo/MSI > .packed_data.lst
awk '{print $2}' .packed_data.lst > .packed_data.name
rm .packed_data.lst
for name in `cat .packed_data.name`
do
	echo $name
	git filter-branch -f --index-filter "git rm -r --cached --ignore-unmatch $name" -- --all
	break
done
rm .packed_data.name
rm -Rf .git/refs/original
git reflog expire --expire=now --all
git gc --aggressive --prune=now
