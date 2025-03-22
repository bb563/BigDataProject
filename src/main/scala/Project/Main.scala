package Project

import org.apache.spark.sql.SparkSession
import scala.io.StdIn
import scala.util.{Try, Success, Failure}

object Main {
  def main(args: Array[String]): Unit = {
    // Initialize SparkSession
    val spark = SparkSession.builder()
      .appName("Disease Analysis")
      .master("local[*]")
      .getOrCreate()

    // Paths to extended data files
    val patientsPath = "data_demo_3/patients_vn_extended_with_country.csv"
    val diseaseGroupsPath = "data_demo_3/disease_groups_vn_extended.csv"

    // Create DiseaseAnalysis instance
    val analysis = new DiseaseAnalysis(spark, patientsPath, diseaseGroupsPath)

    // Display menu and get user input with error handling
    while (true) {
      try {
        println("\n=== Disease Analysis Menu ===")
        println("1. Top 10 Incurable Diseases")
        println("2. Predict Number of Patients for Top Disease")
        println("3. Causes of Top Disease")
        println("4. Age and Gender Distribution of Top Disease")
        println("5. Prevalence of Top Disease by Age Group")
        println("6. Patient Demographics")
        println("7. Risk Factors Analysis")
        println("8. High-Risk Patient Populations")
        println("9. Exit")
        print("Enter your choice (1-9): ")

        // Use Try to safely parse the input
        val choice = Try(StdIn.readLine().trim.toInt) match {
          case Success(value) => value
          case Failure(_) =>
            println("Invalid input! Please enter a number between 1 and 9.")
            -1
        }

        choice match {
          case 1 => analysis.top10IncurableDiseases()
          case 2 => analysis.predictTopDiseaseCount()
          case 3 => analysis.causesOfTopDisease()
          case 4 => analysis.ageGenderDistribution()
          case 5 => analysis.ageGroupPrevalence()
          case 6 => analysis.patientDemographics()
          case 7 => analysis.riskFactorsAnalysis()
          case 8 => analysis.highRiskPatients()
          case 9 =>
            println("Exiting program...")
            spark.stop()
            System.exit(0)
          case _ if choice != -1 => println("Invalid choice! Please enter a number between 1 and 9.")
          case _ => // Already handled in Try block
        }
      } catch {
        case e: Exception =>
          println(s"Error occurred: ${e.getMessage}")
          println("Please try again.")
      }
    }
  }
}