import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions


object MyApp extends App {

  println("Hello world from scala");

  val filePath = getClass.getResource("report_exec_times.csv").getPath
  val spark = SparkSession.builder.appName("Simple Application")
    .master("local").getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  val csvDF = spark.read.format("csv").option("header", "true").load(filePath)

  val dataset1 = csvDF.withColumn("report_params", csvDF.col("report_params").cast(IntegerType))
  val dataset = dataset1.withColumn("exec_time", dataset1.col("exec_time").cast(DoubleType))

  //15 report_id and day_part are categorical features. This means we need to encode these two attributes
  val report_id = dataset.col("report_id")
  val day_part = dataset.col("day_part")

  //16 Encoding categorical attributes (creating as many columns as there are unique values
  // and assigning 1 for the column from current row value)
  def categorize(index: Int) = udf(
    (reportId: Int) => if (reportId == index) 1.0 else 0.0)

  val newDF = dataset.withColumn("report_1", categorize(1)(col("report_id")))
    .withColumn("report_2", categorize(2)(col("report_id")))
    .withColumn("report_3", categorize(3)(col("report_id")))
    .withColumn("report_4", categorize(4)(col("report_id")))
    .withColumn("report_5", categorize(5)(col("report_id")))
    .withColumn("day_morning", categorize(1)(col("day_part")))
    .withColumn("day_midday", categorize(2)(col("day_part")))
    .withColumn("day_afternoon", categorize(3)(col("day_part")))
    .drop("report_id").drop("day_part")


  newDF.show
  //todo: Spark doesn't implement tail() - streaming challenge, it would be inefficient

  // Splitting training dataset into train (80%) and test data
  val splits = newDF.randomSplit(Array(0.8,0.2), 0)
  val trainDF = splits.apply(0)
  val testDF = splits.apply(1)

  def shape(df:DataFrame) = (df.count(), df.columns.length)
  println(shape(trainDF))
  println(shape(testDF))

  // Describe train dataset, without target feature - exec_time.
  // Mean and std will be used to normalize training data


  val trainStats = trainDF.describe().drop("exec_time")
  trainStats.show

  //transpose
  //https://stackoverflow.com/questions/40892459/spark-transpose-dataframe-without-aggregating
  val kv = explode(array(trainStats.columns.tail.map {
    c => struct(lit(c).alias("k"), col(c).alias("v"))
  }: _*))
  val transposed = trainStats
    .withColumn("kv", kv)
    .select(col("summary"), col("kv.k"), col("kv.v"))
    .groupBy(col("k"))
    .pivot("summary")
    .agg(first(col("v")))
    .orderBy(col("k"))
    .withColumnRenamed("k", "vals")

  transposed.show

  //Remove exec_time feature from training data and keep it as a target for both training and testing

  val trainLabels = trainDF.col("exec_time")
  val trainDF2 = trainDF.drop("exec_time")
  val test_labels = testDF.col("exec_time")
  val testDF2 = testDF.drop("exec_time")
  // Neural network learns better, when data is normalized (features look similar to each other)
  def normalize( df:DataFrame): DataFrame =
  {
    val assembler = new VectorAssembler()
      .setInputCols(df.columns)
      .setOutputCol("features")

    val pipelineVectorAssembler = new Pipeline().setStages(Array(assembler))
    val withFeatures = pipelineVectorAssembler.fit(df).transform(df)
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)
    scaler.fit(withFeatures).transform(withFeatures)
  }
  val normedTrainDF = normalize(trainDF2)
  val normedTestDF = normalize(testDF2)
  normedTrainDF.show

//  val features = normedTrainDF.select(collect_list("scaledFeatures"))
//    .first()
//    .getList[DenseVector](0)
//  val arrData = features.toArray(new Array[DenseVector](features.size()))
//  println(arrData.length);

  val features = normedTrainDF.select("scaledFeatures").collect().map( f =>
  {
    f.get(0).asInstanceOf[DenseVector].values
  })

  val dataNdArray = Nd4j.create(features)

    val labels = trainDF.select("exec_time").collect().map(f =>
      Array(f.getDouble(0))
    )
      //.select(collect_list("exec_time")).first().getList[Double](0)
  //val arrLabels = labels.toArray(new Array[AnyRef](labels.size()))

  val labelsNdArray = Nd4j.create(labels)

  val conf = new NeuralNetConfiguration.Builder()
    .seed(0)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(new Sgd(0.001))
    .list
    .layer(0, new DenseLayer.Builder()
      .units(50)
      .nIn(dataNdArray.columns())
      .activation(Activation.SIGMOID)
      //.weightInit(WeightInit.SIGMOID_UNIFORM)
      .build
    )
    .layer(1, new DenseLayer.Builder()
      .units(50)
      .activation(Activation.SIGMOID).build)
    .layer( 2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.SIGMOID)
      .nOut(1)
      .build())
    .build

  val network = new MultiLayerNetwork(conf)
  network.init()

  // pass a training listener that reports score every 10 iterations
  val eachIterations = 5
  network.addListeners(new ScoreIterationListener(eachIterations))
  network.setEpochCount(1000)

  network.fit(dataNdArray, labelsNdArray)

  //val eval = network.evaluate();

  spark.stop()
}



//  val conf = new NeuralNetConfiguration.Builder()
//    .seed(rngSeed)
//    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//    .updater(new Adam())
//    .l2(1e-4)
//    .list()
//    .layer(new DenseLayer.Builder()
//      .nIn(numRows * numColumns) // Number of input datapoints.
//      .nOut(1000) // Number of output datapoints.
//      .activation(Activation.RELU) // Activation function.
//      .weightInit(WeightInit.XAVIER) // Weight initialization.
//      .build())
//    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//      .nIn(1000)
//      .nOut(outputNum)
//      .activation(Activation.SOFTMAX)
//      .weightInit(WeightInit.XAVIER)
//      .build())
//    .pretrain(false).backprop(true)
//    .build()


