ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.12"

lazy val root = (project in file("."))
  .settings(
    name := "FinalProject2"
  )
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.5.4"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.4"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.4.1"