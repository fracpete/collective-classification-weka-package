How to make a release
=====================

Preparation
-----------

* Change the artifact ID in `pom.xml` to today's date, e.g.:

  ```
  2015.2.27-SNAPSHOT
  ```

* Update the version, date and URL in `Description.props` to reflect new
  version, e.g.:

  ```
  Version=2015.2.27
  Date=2015-02-27
  PackageURL=https://github.com/fracpete/collective-classification-weka-package/releases/download/v2015.2.27/collective-classification-2015.2.27.zip
  ```

Weka package
------------

* Commit/push all changes

* Run the following command to generate the package archive for version `2015.2.27`:

  ```
  ant -f build_package.xml -Dpackage=collective-classification-2015.2.27 clean make_package
  ```

* Create a release tag on github (v2015.2.27)
* add release notes
* upload package archive from `dist`


Maven
-----

* Run the following command to deploy the artifact:

  ```
  mvn release:clean release:prepare release:perform
  ```

* log into https://oss.sonatype.org and close/release artifacts

* After successful deployment, push the changes out:

  ```
  git push
  ````

