name := "ReportTimesMLScala"

version := "0.1"

scalaVersion := "2.12.10"



ThisBuild / useCoursier := false

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.4",
  "org.apache.spark" %% "spark-sql" % "2.4.4",
  "org.apache.spark" %% "spark-mllib" % "2.4.4",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta5",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta5"
)
