branchname=$1
if ! git checkout $branchname
then
    echo >&2 "branch $branchname does not exist"
    exit 1
fi
git checkout master
git tag archive/$branchname $branchname
git push --tags
echo "tagged branch $branchname as archive"
git branch -D $branchname
git push origin --delete $branchname