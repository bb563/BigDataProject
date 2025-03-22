package Project

import org.apache.spark.sql.{SparkSession, Dataset, Encoders}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

// Cập nhật case class để thêm country
case class Patient(
                    birthDate: String,
                    gender: String,
                    disease: String,
                    subgroupId: String,
                    region: String,
                    incomeLevel: String,
                    smokingStatus: String,
                    bmi: Double,
                    comorbidity: Int,
                    country: String
                  )

case class DiseaseGroup(
                         disease: String,
                         subgroupId: String,
                         weight: Double,
                         causeDescription: String,
                         riskFactor: String
                       )

class DiseaseAnalysis(spark: SparkSession, patientsPath: String, diseaseGroupsPath: String) {
  import spark.implicits._

  // Định nghĩa UDF tính tuổi
  private val ageUDF = udf(AgeUtils.calculateAge _)

  // Load data as Datasets với cấu trúc mới
  private val patientsDS: Dataset[Patient] = spark.read
    .option("header", "true")
    .csv(patientsPath)
    .withColumnRenamed("patient_birth_date", "birthDate")
    .withColumnRenamed("patient_gender", "gender")
    .withColumnRenamed("disease_name", "disease")
    .withColumnRenamed("subgroup_id", "subgroupId")
    .withColumnRenamed("income_level", "incomeLevel")
    .withColumnRenamed("smoking_status", "smokingStatus")
    .withColumn("bmi", col("bmi").cast("double"))
    .withColumn("comorbidity", col("comorbidity").cast("int"))
    .as[Patient]

  private val diseaseGroupsDS: Dataset[DiseaseGroup] = spark.read
    .option("header", "true")
    .csv(diseaseGroupsPath)
    .withColumnRenamed("disease_name", "disease")
    .withColumnRenamed("subgroup_id", "subgroupId")
    .withColumn("weight", col("weight").cast("double"))
    .withColumnRenamed("cause_description", "causeDescription")
    .withColumnRenamed("risk_factor", "riskFactor")
    .as[DiseaseGroup]

  // Chức năng 1: Top 10 bệnh nan y phổ biến nhất
  def top10IncurableDiseases(): Unit = {
    println("Top 10 Incurable Diseases:")
    patientsDS.groupBy($"disease")
      .agg(count("*").as("patient_count"))
      .orderBy(desc("patient_count"))
      .limit(10)
      .show(truncate = false)
  }

  // Chức năng 2: Dự đoán số người mắc bệnh top 1 bằng MLlib (Linear Regression)
  def predictTopDiseaseCount(): Unit = {
    val topDisease = patientsDS.groupBy($"disease")
      .agg(count("*").as("patient_count"))
      .orderBy(desc("patient_count"))
      .limit(1)
      .select("disease")
      .first()
      .getString(0)

    val topDiseaseData = patientsDS.filter($"disease" === topDisease)
      .withColumn("year", substring($"birthDate", 1, 4).cast("double"))
      .groupBy($"year")
      .agg(count("*").as("patient_count"))
      .filter($"year".isNotNull)
      .orderBy($"year")

    val assembler = new VectorAssembler()
      .setInputCols(Array("year"))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val dataWithFeatures = assembler.transform(topDiseaseData)
      .select($"features", $"patient_count".as("label"))

    val Array(trainingData, testData) = dataWithFeatures.randomSplit(Array(0.8, 0.2), seed = 42)

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(10)
      .setRegParam(0.01)
      .setElasticNetParam(0.0)

    val model = lr.fit(trainingData)

    val currentYear = 2025
    val futureYears = (currentYear + 1 to currentYear + 5).map(_.toDouble)
    val futureData = spark.createDataset(futureYears.map(year => (year)))
      .toDF("year")

    val futureDataWithFeatures = assembler.transform(futureData)

    val futurePredictions = model.transform(futureDataWithFeatures)
      .select($"year", $"prediction".cast("long").as("predicted_count"))

    val currentCount = topDiseaseData.agg(sum("patient_count")).first().getLong(0)
    println(s"Prediction for the most common disease '$topDisease':")
    println(s"Current count (up to 2025): $currentCount")
    println("Predicted counts for the next 5 years:")
    futurePredictions.show(truncate = false)
  }

  // Chức năng 3: Nguyên nhân dẫn đến bệnh top 1
  def causesOfTopDisease(): Unit = {
    val topDisease = patientsDS.groupBy($"disease")
      .agg(count("*").as("patient_count"))
      .orderBy(desc("patient_count"))
      .limit(1)
      .select("disease")
      .first()
      .getString(0)

    val causesFromGroups = diseaseGroupsDS.filter($"disease" === topDisease)
      .select($"subgroupId", $"causeDescription", $"weight")

    val patientCauses = patientsDS.filter($"disease" === topDisease)
      .groupBy($"subgroupId")
      .agg(count("*").as("patient_count"))

    val totalPatients = patientsDS.filter($"disease" === topDisease).count()
    val causes = causesFromGroups
      .join(patientCauses, Seq("subgroupId"), "left_outer")
      .withColumn("actual_percentage", (col("patient_count") * 100.0) / totalPatients)
      .withColumn("estimated_percentage", col("weight") * 100)
      .select(
        $"subgroupId",
        $"causeDescription",
        $"estimated_percentage".as("estimated_contribution_%"),
        $"actual_percentage".as("actual_contribution_%")
      )
      .orderBy(desc("actual_contribution_%"))

    println(s"Possible causes of the most common disease '$topDisease' (Total patients: $totalPatients):")
    println("Causes with estimated (from literature) and actual (from patient data) contributions:")
    causes.show(truncate = false)
  }

  // Chức năng 4: Phân bố tuổi và giới tính của bệnh top 1
  def ageGenderDistribution(): Unit = {
    val topDisease = patientsDS.groupBy($"disease")
      .agg(count("*").as("patient_count"))
      .orderBy(desc("patient_count"))
      .limit(1)
      .select("disease")
      .first()
      .getString(0)

    val distribution = patientsDS.filter($"disease" === topDisease)
      .withColumn("age", ageUDF($"birthDate"))
      .groupBy($"gender", $"age")
      .agg(count("*").as("count"))
      .orderBy($"age", $"gender")

    println(s"Age and Gender Distribution for '$topDisease':")
    distribution.show(truncate = false)
  }

  // Chức năng 5: Tỷ lệ mắc bệnh top 1 theo nhóm tuổi
  def ageGroupPrevalence(): Unit = {
    val topDisease = patientsDS.groupBy($"disease")
      .agg(count("*").as("patient_count"))
      .orderBy(desc("patient_count"))
      .limit(1)
      .select("disease")
      .first()
      .getString(0)

    val totalTopDiseasePatients = patientsDS.filter($"disease" === topDisease).count()

    val ageGroups = patientsDS.filter($"disease" === topDisease)
      .withColumn("age", ageUDF($"birthDate"))
      .withColumn("age_group",
        when($"age" < 20, "0-19")
          .when($"age".between(20, 39), "20-39")
          .when($"age".between(40, 59), "40-59")
          .when($"age".between(60, 79), "60-79")
          .otherwise("80+"))
      .groupBy($"age_group")
      .agg(count("*").as("disease_count"))
      .withColumn("prevalence_percentage", (col("disease_count") * 100.0) / totalTopDiseasePatients)
      .orderBy($"age_group")

    println(s"Prevalence of '$topDisease' by Age Group (percentage out of $totalTopDiseasePatients patients with this disease):")
    ageGroups.show(truncate = false)
  }

  // Chức năng 6: Phân tích đặc điểm dân số học (Patient Demographics) - Thêm country
  def patientDemographics(): Unit = {
    val topDisease = patientsDS.groupBy($"disease")
      .agg(count("*").as("patient_count"))
      .orderBy(desc("patient_count"))
      .limit(1)
      .select("disease")
      .first()
      .getString(0)

    val totalPatients = patientsDS.filter($"disease" === topDisease).count()

    // Phân tích theo country
    val byCountry = patientsDS.filter($"disease" === topDisease)
      .groupBy($"country")
      .agg(count("*").as("patient_count"))
      .withColumn("percentage", (col("patient_count") * 100.0) / totalPatients)
      .orderBy(desc("patient_count"))

    // Phân tích theo region
    val byRegion = patientsDS.filter($"disease" === topDisease)
      .groupBy($"region")
      .agg(count("*").as("patient_count"))
      .withColumn("percentage", (col("patient_count") * 100.0) / totalPatients)
      .orderBy(desc("patient_count"))

    // Phân tích theo income_level
    val byIncome = patientsDS.filter($"disease" === topDisease)
      .groupBy($"incomeLevel")
      .agg(count("*").as("patient_count"))
      .withColumn("percentage", (col("patient_count") * 100.0) / totalPatients)
      .orderBy(desc("patient_count"))

    // Phân tích theo gender
    val byGender = patientsDS.filter($"disease" === topDisease)
      .groupBy($"gender")
      .agg(count("*").as("patient_count"))
      .withColumn("percentage", (col("patient_count") * 100.0) / totalPatients)
      .orderBy(desc("patient_count"))

    println(s"Demographics for the most common disease '$topDisease' (Total patients: $totalPatients):")
    println("By Country:")
    byCountry.show(truncate = false)
    println("By Region:")
    byRegion.show(truncate = false)
    println("By Income Level:")
    byIncome.show(truncate = false)
    println("By Gender:")
    byGender.show(truncate = false)
  }

  // Chức năng 7: Yếu tố nguy cơ liên quan đến bệnh mãn tính (Risk Factors)
  def riskFactorsAnalysis(): Unit = {
    val topDisease = patientsDS.groupBy($"disease")
      .agg(count("*").as("patient_count"))
      .orderBy(desc("patient_count"))
      .limit(1)
      .select("disease")
      .first()
      .getString(0)

    val totalPatients = patientsDS.filter($"disease" === topDisease).count()

    // Lấy yếu tố nguy cơ từ diseaseGroupsDS
    val riskFactorsFromGroups = diseaseGroupsDS.filter($"disease" === topDisease)
      .select($"subgroupId", $"riskFactor", $"weight")

    // Đếm số bệnh nhân theo subgroupId và smoking_status
    val patientRiskFactors = patientsDS.filter($"disease" === topDisease)
      .groupBy($"subgroupId", $"smokingStatus")
      .agg(count("*").as("patient_count"))

    // Join và tính tỷ lệ
    val riskAnalysis = riskFactorsFromGroups
      .join(patientRiskFactors, Seq("subgroupId"), "left_outer")
      .withColumn("actual_percentage", (col("patient_count") * 100.0) / totalPatients)
      .withColumn("estimated_percentage", col("weight") * 100)
      .select(
        $"subgroupId",
        $"riskFactor",
        $"smokingStatus",
        $"estimated_percentage".as("estimated_contribution_%"),
        $"actual_percentage".as("actual_contribution_%")
      )
      .orderBy(desc("actual_contribution_%"))

    println(s"Risk Factors for the most common disease '$topDisease' (Total patients: $totalPatients):")
    println("Risk factors with estimated (from literature) and actual (from patient data) contributions:")
    riskAnalysis.show(truncate = false)
  }

  // Chức năng 8: Xác định nhóm bệnh nhân nguy cơ cao (High-Risk Patient Populations)
  def highRiskPatients(): Unit = {
    val topDisease = patientsDS.groupBy($"disease")
      .agg(count("*").as("patient_count"))
      .orderBy(desc("patient_count"))
      .limit(1)
      .select("disease")
      .first()
      .getString(0)

    // Xác định bệnh nhân nguy cơ cao: BMI > 30 (béo phì), có hút thuốc, và có bệnh đồng mắc
    val highRisk = patientsDS.filter($"disease" === topDisease)
      .filter($"bmi" > 30 && $"smokingStatus" === "Yes" && $"comorbidity" >= 1)
      .withColumn("age", ageUDF($"birthDate"))
      .groupBy($"region", $"incomeLevel", $"gender")
      .agg(
        count("*").as("high_risk_count"),
        avg("bmi").as("avg_bmi"),
        avg("comorbidity").as("avg_comorbidity")
      )
      .orderBy(desc("high_risk_count"))

    println(s"High-Risk Patient Populations for '$topDisease':")
    println("Criteria: BMI > 30, Smokes, and has at least 1 comorbidity")
    highRisk.show(truncate = false)
  }
}