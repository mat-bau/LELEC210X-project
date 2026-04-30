/* USER CODE BEGIN Header */
/**
  ****************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ****************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "aes.h"
#include "dma.h"
#include "spi.h"
#include "tim.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "arm_math.h"
#include "adc_dblbuf.h"
#include "retarget.h"
#include "s2lp.h"
#include "spectrogram.h"
#include "eval_radio.h"
#include "packet.h"
#include "config.h"
#include "utils.h"
#include "usart.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
volatile uint8_t btn_press = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == B1_Pin) {
        btn_press = 1;
    }
    else if (GPIO_Pin == RADIO_INT_Pin) {
        S2LP_IRQ_Handler();
    }
}

void run(void)
{
    ADC_StartListening();

    while (1)
    {
        __WFI();
    }
}

/* USER CODE END 0 */

int main(void)
{
    HAL_Init();
    SystemClock_Config();

    MX_GPIO_Init();
    MX_DMA_Init();
    MX_SPI1_Init();
    MX_TIM3_Init();
    MX_ADC1_Init();
    MX_AES_Init();

    /* USER CODE BEGIN 2 */
    if (ENABLE_UART) {
        MX_LPUART1_UART_Init();
    }

    RetargetInit(&hlpuart1);
    DEBUG_PRINT("\r\n========================================\r\n");
    DEBUG_PRINT("STM32 Spectrogram System\r\n");
    DEBUG_PRINT("========================================\r\n");

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    Spectrogram_Init();

#if ENABLE_RADIO
    HAL_StatusTypeDef err = S2LP_Init(&hspi1);
    if (err) {
        DEBUG_PRINT("[S2LP] Error: %u\r\n", err);
        Error_Handler();
    } else {
        DEBUG_PRINT("[S2LP] Init OK\r\n");
    }
#endif

    if (HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED) != HAL_OK) {
        DEBUG_PRINT("ADC calibration error\r\n");
        Error_Handler();
    }

    if (HAL_TIM_Base_Start(&htim3) != HAL_OK) {
        DEBUG_PRINT("Timer error\r\n");
        Error_Handler();
    }

    DEBUG_PRINT("HCLK: %lu Hz\r\n", HAL_RCC_GetHCLKFreq());
    DEBUG_PRINT("TIM3 freq: %lu Hz\r\n", HAL_RCC_GetHCLKFreq() / (htim3.Init.Prescaler + 1) / (htim3.Init.Period + 1));
    DEBUG_PRINT("\r\nSystem ready!\r\n");
    DEBUG_PRINT("========================================\r\n");
    /* USER CODE END 2 */

#if (RUN_CONFIG == MAIN_APP)
    run();
#elif (RUN_CONFIG == EVAL_RADIO)
    eval_radio();
#endif

    while (1) {}
}

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
        Error_Handler();

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
    RCC_OscInitStruct.MSIState = RCC_MSI_ON;
    RCC_OscInitStruct.MSICalibrationValue = 0;
    RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_5;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
        Error_Handler();

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                                |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_MSI;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
        Error_Handler();
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

void Error_Handler(void)
{
    __disable_irq();
    while (1) {
        HAL_GPIO_WritePin(GPIOB, LD3_Pin, GPIO_PIN_SET);
        for (volatile int i = 0; i < SystemCoreClock / 200; i++);
        HAL_GPIO_WritePin(GPIOB, LD3_Pin, GPIO_PIN_RESET);
        for (volatile int i = 0; i < SystemCoreClock / 200; i++);
    }
}

#ifdef  USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif /* USE_FULL_ASSERT */
