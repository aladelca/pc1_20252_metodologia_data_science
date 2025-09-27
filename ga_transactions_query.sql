-- Google Analytics Universal Analytics Transaction Analysis Query
-- Shows transactions at date/product level with session and product information

SELECT
  -- Date Information
  date AS transaction_date,
  PARSE_DATE('%Y%m%d', date) AS parsed_date,

  -- Transaction Information
  hits.transaction.transactionId AS transaction_id,
  hits.transaction.transactionRevenue / 1000000 AS transaction_revenue_usd,
  hits.transaction.transactionTax / 1000000 AS transaction_tax_usd,
  hits.transaction.transactionShipping / 1000000 AS transaction_shipping_usd,
  hits.transaction.affiliation AS transaction_affiliation,
  hits.transaction.currencyCode AS currency_code,

  -- Product Information
  product.productSKU AS product_sku,
  product.v2ProductName AS product_name,
  product.v2ProductCategory AS product_category,
  product.productBrand AS product_brand,
  product.productVariant AS product_variant,
  product.productQuantity AS product_quantity,
  product.productPrice / 1000000 AS product_price_usd,
  product.productRevenue / 1000000 AS product_revenue_usd,
  product.isImpression AS is_impression,
  product.isClick AS is_click,

  -- Promotion Information
  promotions.promoId AS promo_id,
  promotions.promoName AS promo_name,
  promotions.promoCreative AS promo_creative,
  promotions.promoPosition AS promo_position,

  -- Session Information
  fullVisitorId AS visitor_id,
  visitId AS session_id,
  visitNumber AS session_number,
  visitStartTime AS session_start_time,
  totals.visits AS total_visits,
  totals.hits AS total_hits,
  totals.pageviews AS total_pageviews,
  totals.timeOnSite AS time_on_site_seconds,
  totals.bounces AS bounces,
  totals.newVisits AS new_visits,

  -- Traffic Source Information
  trafficSource.source AS traffic_source,
  trafficSource.medium AS traffic_medium,
  trafficSource.campaign AS campaign,
  trafficSource.keyword AS keyword,
  trafficSource.adContent AS ad_content,
  trafficSource.referralPath AS referral_path,
  trafficSource.isTrueDirect AS is_true_direct,
  channelGrouping AS channel_grouping,

  -- Device Information
  device.browser AS browser,
  device.browserVersion AS browser_version,
  device.operatingSystem AS operating_system,
  device.operatingSystemVersion AS os_version,
  device.isMobile AS is_mobile,
  device.mobileDeviceBranding AS mobile_device_brand,
  device.mobileDeviceModel AS mobile_device_model,
  device.deviceCategory AS device_category,
  device.language AS device_language,
  device.screenResolution AS screen_resolution,

  -- Geographic Information
  geoNetwork.continent AS continent,
  geoNetwork.subContinent AS sub_continent,
  geoNetwork.country AS country,
  geoNetwork.region AS region,
  geoNetwork.metro AS metro,
  geoNetwork.city AS city,
  geoNetwork.networkDomain AS network_domain,

  -- AdWords Information (if applicable)
  trafficSource.adwordsClickInfo.campaignId AS adwords_campaign_id,
  trafficSource.adwordsClickInfo.adGroupId AS adwords_adgroup_id,
  trafficSource.adwordsClickInfo.creativeId AS adwords_creative_id,
  trafficSource.adwordsClickInfo.criteriaId AS adwords_criteria_id,
  trafficSource.adwordsClickInfo.gclId AS gclid,
  trafficSource.adwordsClickInfo.adNetworkType AS ad_network_type,

  -- Hit Level Information
  hits.hitNumber AS hit_number,
  hits.time AS hit_time_ms,
  hits.hour AS hit_hour,
  hits.minute AS hit_minute,
  hits.isInteraction AS is_interaction,
  hits.isEntrance AS is_entrance,
  hits.isExit AS is_exit,
  hits.referer AS referer,

  -- Page Information (for transaction hits)
  hits.page.pagePath AS page_path,
  hits.page.hostname AS hostname,
  hits.page.pageTitle AS page_title,

  -- Entrance and Exit Pages for the session
  FIRST_VALUE(CASE WHEN CAST(hits.isEntrance AS STRING) = 'true' THEN hits.page.pagePath END IGNORE NULLS)
    OVER (PARTITION BY fullVisitorId, visitId ORDER BY hits.hitNumber) AS entrance_page_path,

  FIRST_VALUE(CASE WHEN CAST(hits.isExit AS STRING) = 'true' THEN hits.page.pagePath END IGNORE NULLS)
    OVER (PARTITION BY fullVisitorId, visitId ORDER BY hits.hitNumber DESC) AS exit_page_path,

  -- Social Engagement
  socialEngagementType AS social_engagement_type

FROM
  `your-project.your-dataset.ga_sessions_*` AS sessions,
  UNNEST(hits) AS hits,
  UNNEST(hits.product) AS product
LEFT JOIN UNNEST(hits.promotion) AS promotions

WHERE
  -- Filter for transaction hits only
  hits.eCommerceAction.action_type = '6'  -- Purchase action
  AND hits.transaction.transactionId IS NOT NULL
  AND product.productSKU IS NOT NULL

  -- Date filter (adjust as needed)
  AND _TABLE_SUFFIX BETWEEN '20170801' AND '20171231'

ORDER BY
  transaction_date DESC,
  transaction_id,
  product_sku;
