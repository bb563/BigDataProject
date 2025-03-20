package Project

object AgeUtils {
  def calculateAge(birthDate: String): Int = {
    val birthYear = birthDate.split("-")(0).toInt
    val currentYear = 2025 // Giả định năm hiện tại là 2025
    currentYear - birthYear
  }
}
