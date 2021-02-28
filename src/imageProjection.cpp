#include "utility.h"
#include "lio_sam/cloud_info.h"

// Velodyne点云点结构体构造
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY; // xyz和强度intensity
    uint16_t ring; // 线数
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 字符对齐
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

// Ouster 略
struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

// 缓存队列长度，imu
const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;


public:
    ImageProjection():
    deskewFlag(0)
    {
        // 订阅话题进入回调函数：imu；激光点云；地图优化后发布的里程（增量式）
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布话题：去畸变的点云；点云信息（自定义消息 msg/cloud_info.msg）
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);
        // 分配内存
        allocateMemory();
        // 初始化参数
        resetParameters();
        // PCL控制台输出ERROR信息
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    /*根据激光雷达线数分配内存（config/params.yaml内定义N_SCAN和Horizon_SCAN）*/
    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters(); //repeat
    }

    /*初始化操作*/
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    /*imu回调*/
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        // 将imu坐标系下数据转到雷达坐标系下
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        // 锁在{～}内生效
        std::lock_guard<std::mutex> lock1(imuLock);
        // 入队尾
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    /*里程计回调*/
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        // 锁存
        std::lock_guard<std::mutex> lock2(odoLock);
        // 入队尾
        odomQueue.push_back(*odometryMsg);
    }

    /*点云回调*/
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // 检查是否够两帧点云
        if (!cachePointCloud(laserCloudMsg))
            return;
        // 检查是否有imu数据，第一个imu小于雷达当前帧时间戳, 最后一个imu小于下一帧时间戳
        if (!deskewInfo())
            return;

        projectPointCloud();

        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    /*缓存点云到队列，作一些检查，线束、时间通道等*/
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // 入队尾
        cloudQueue.push_back(*laserCloudMsg);
        // 点云太少直接return  至少两帧数据
        if (cloudQueue.size() <= 2)
            return false;
        // 点云消息的队列先进先出
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        // 得到点云
        if (sensor == SensorType::VELODYNE)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header; // 包头
        timeScanCur = cloudHeader.stamp.toSec(); // 当前帧时间戳
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time; // 最后扫过的点

        // check dense flag 检查有没有无效点
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel 检查线束通道
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time 检查时间通道，去畸变须用
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    /*保证imu有数据且包含扫描帧时间，然后利用imu和odom作去畸变*/
    bool deskewInfo()
    {
        // 锁： 确保数据都进来了
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // 满足时间：第一个imu小于雷达当前帧时间戳, 最后一个imu小于下一帧时间戳
        // 保证imu有数据且包含扫描帧时间
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }
        // imu去畸变
        imuDeskewInfo();
        // 里程计去畸变
        odomDeskewInfo();

        return true;
    }

    /*下一帧点云前，判断imu数据是否可用，计算imu累计的角度*/
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false; // 点云信息标志imu不可用，后面mapOptmization需用
        // 直到imu数据的时间戳到当前帧前0.01s以内，舍弃较旧的imu数据
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }
        // imu为空，不作去畸变
        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i]; // 取出imu
            double currentImuTime = thisImuMsg.header.stamp.toSec(); // 时间戳

            // get roll, pitch, and yaw estimation for this scan
            // imu时间戳小于当前帧点云时间戳，坐标转换后，直接赋到了cloudInfo中
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
            // 大于下一帧退出
            if (currentImuTime > timeScanEnd + 0.01)
                break;
            // 初始化为0
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }
            // get angular velocity
            // 从imu中获得角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);
            // integrate rotation
            // 把角速度和时间间隔积分出转角  用于后续的去畸变
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur; // 指向最后一个转角积分

        if (imuPointerCur <= 0) // 无转角积分，退出
            return;

        cloudInfo.imuAvailable = true; // 点云信息标志imu可用
    }

    /*获取当前帧起始的里程计位姿估计，计算与末尾时刻之间的6degree增量用于去畸变*/
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false; // 点云信息标志里程计不可用，mapOptmization中需用
        // 直到里程计数据的时间戳到当前帧前0.01s以内，舍弃较旧的里程计数据
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }
        // 无里程计数据
        if (odomQueue.empty())
            return;
        // 时间戳大于下一帧退出
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;
        // 获取当前帧起始时刻的里程计消息
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];
            // 在起始时刻前0.1s以内
            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation); //四元数 to tf

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw); // 位姿角

        // Initial guess used in mapOptimization
        // 利用里程计消息生成初始位姿估计， 存在cloudInfo中
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true; // 里程计可用

        // get end odometry at the end of the scan
        odomDeskewFlag = false; // 不进行去畸变
        // 获取当前帧末尾之后的里程计信息，用于去畸变（运动补偿）
        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];
            // 当前帧末尾之后的
            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }
        // 位姿协方差矩阵判断
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;
        // 初始里程计变换
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 末尾里程计变换
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 获得当前帧起始与末尾时刻之间的变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
        // 通过这个变换获得6degree增量值
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);
        // 去畸变flag
        odomDeskewFlag = true;
    }

    /*利用imu积分；根据激光点的时间信息 获得该时刻的旋转变化量*/
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0; // init
        // 遍历寻找对应的imu数据
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        // imuFront < point < imuBack，或 point < imuFront < imuBack
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront]; // 直接用积分
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else { // imuFront < imuBack < point
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack; // 用倒数两个积分
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack; // 按比例估计
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
            // imuPointerBack < imuPointerFront < point
            // |<-------------ratioFront------------->|
            //                  |<-----ratioBack----->|
        }
    }

    /*利用了里程计增量；；作者注释掉了该功能函数，在高速移动下可能有用，低速下提升很小*/
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;
        // 当前激光点的比例
        // float ratio = relTime / (timeScanEnd - timeScanCur);
        // 按比例估计odom增量
        // timeScanCur <= startOdomMsg < timeScanEnd <= endOdomMsg
        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    /*传入激光点PointType(XYZI)和相对时间戳，利用imu和odom作运动补偿*/
    PointType deskewPoint(PointType *point, double relTime)
    {
        // 点云无时间通道 或 imu不可用，退出
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;
        // 激光点的绝对时间戳
        double pointTime = timeScanCur + relTime;
        // 利用imu数据补偿，旋转量
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);
        // 利用odom数据补偿，平移量
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);
        // 第一次收到数据
        if (firstPointFlag == true)
        {   //计算起始变换矩阵，并取逆
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start 把点投影到每一帧点云的起始时刻
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;
        // 得到去畸变的点云
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    /*将点云投影到深度图像，按行列保存（同LOAM）*/
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size(); // 点云数
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint; // XYZI
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            // 与光心的距离，太远太近都不要
            float range = pointDistance(thisPoint); // 与原点距离
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;
            // 激光点垂直方向ring号（行序号）在0-N_SCAN内
            // 以velodyne vlp16为例：从下往上计数，-15度记为初始线，第0线，一共16线(N_SCAN =16)
            int rowIdn = laserCloudIn->points[i].ring; // 
            // 行线数不正确
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 下采样
            if (rowIdn % downsampleRate != 0)
                continue;
            // 计算激光点水平方向上的角度
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            // 角分辨率 360度/Horizon_SCAN
            static float ang_res_x = 360.0/float(Horizon_SCAN);
            // 计算激光点属于哪一列
            // 以velodyne vlp16为例：从负y为0线，逆时针，一共1800线(Horizon_SCAN =1800)
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN) // 超过了一圈
                columnIdn -= Horizon_SCAN;
            // 列线数不正确
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            // 没有初始化
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            // 去畸变
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            // 存储range矩阵
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            // 激光点的索引值
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint; // fullCloud存储去畸变点云
        }
    }

    /*提取点云*/
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        // 由行列存储变成列表存储
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }

    /*发布去畸变点云和点云信息*/
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
