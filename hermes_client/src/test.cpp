#include "HermesSDK.h"

#include "spdlog/spdlog.h"
#include "concurrentqueue.h"

#include <conio.h>
#include <mutex>
#include <deque>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int16.h>
#include <geometry_msgs/QuaternionStamped.h>

#define _DEBUG
#define _TRACE

class Timer
{
public:
	Timer() {}

	void Log()
	{
		AddSample(TimeNow());
		// LogLastSample(); // only use this in case of needing to export to a CSV file.
	}

	double GetAverageRate()
	{
		std::lock_guard<std::mutex> stampGuard(Lock);
		double dt = Stamps[Stamps.size() - 1] - Stamps[0];
		if (dt == 0)
		{
			return 0;
		}
		else
		{
			return Stamps.size() / dt;
		}
	}
	double GetLargestGap()
	{
		return LargestGap;
	}

	void WriteLog(std::string filename) const
	{
		std::ofstream logfile;
		logfile.open(filename);
		logfile << "#time,rate,gap" << std::endl;
		for (const auto& stampedRate : StampedRateLog)
		{
			logfile << stampedRate.time << "," << stampedRate.rate << "," << stampedRate.gap << std::endl;
		}
		logfile.close();
	}

private:
	struct StampedRate {
		double time;
		double rate;
		double gap;
	};
	std::vector<StampedRate> StampedRateLog;
	std::deque<double> Stamps;
	uint64_t RefTicks = 0;
	bool RefTicksInitialized = false;
	std::mutex Lock;
	int RunningAverageNum = 200;
	double LatestTimeStamp = 0;
	double LatestGap = 0;
	double LargestGap = 0;

	double TimeNow()
	{
		LARGE_INTEGER perfFrequency;
		::QueryPerformanceFrequency(&perfFrequency);
		uint64_t clockFrequency = perfFrequency.QuadPart;
		LARGE_INTEGER nowTimestamp;
		::QueryPerformanceCounter(&nowTimestamp);
		uint64_t nowTicks = nowTimestamp.QuadPart;
		if (!RefTicksInitialized)
		{
			RefTicks = nowTicks;
			RefTicksInitialized = true;
		}
		return static_cast<double> (nowTicks - RefTicks) / static_cast<double>(clockFrequency);
	}

	void AddSample(const double& t)
	{
		std::lock_guard<std::mutex> stampGuard(Lock);
		if (Stamps.size() < RunningAverageNum)
		{
			Stamps.push_back(t);
		}
		else
		{
			Stamps.pop_front();
			Stamps.push_back(t);
		}

		if (LatestTimeStamp == 0)
		{
			LatestTimeStamp = t;
		}
		LatestGap = t - LatestTimeStamp;
		LargestGap = std::max(LatestGap, LargestGap);
		LatestTimeStamp = t;
	}

	void LogLastSample()
	{
		StampedRateLog.push_back(StampedRate({ LatestTimeStamp, GetAverageRate(), LatestGap }));
	}
};

class Sample
{
private:
	Hermes::Protocol::Hardware::DeviceLandscape m_Landscape;
	std::mutex m_Print_mutex; // make sure different threads are not writing/reading the same variables at the same time
	std::mutex m_Landscape_mutex; // make sure different threads are not writing/reading the same variables at the same time

	//ROS
	ros::NodeHandle nh;
	ros::Publisher pub_norm;
	ros::Publisher pub_deg;
	ros::Publisher pub_quat;
	// ros::Subscriber sub;
	
	// map of glove ids -> timers
	std::map<uint64_t, Timer> m_timers;
	std::map<uint64_t, double> m_gloveRates;
	std::map<uint64_t, double> m_gloveGaps;
	std::map<uint64_t, size_t> m_glovePrintOrder;
	const size_t nrOfKeyInfoLines = 4;
	const size_t nrOfDataGloveInfoLines = 7;

	static const size_t fingerCount = 5;
	static const size_t jointPerFingerCount = 3;
	struct RumbleState {
		bool wrist;
		std::array<bool, fingerCount> finger;
	};

	std::map<Hermes::Protocol::HandType, RumbleState> m_rumbleState;
	const size_t timeBetweenHapticsCmds_ms = 20;
	std::chrono::high_resolution_clock::time_point m_timeLastHapticsCmdSent;
	Timer m_hapticsTimer;
	const size_t clientTimerId = 0; // for the overall client timer
	int m_updates_without_data = 0;
	bool m_tabPreviouslyPressed = false;

	// print joint angles as Normalized, Degrees or Quaternions
	enum class JointAnglePrintMode
	{
		Normalized,
		Degrees,
		Quaternions,
		NumberOfAnglePrintModes
	};

	JointAnglePrintMode m_jointDataViewMode = JointAnglePrintMode::Normalized;

	moodycamel::ConcurrentQueue<Hermes::Protocol::Devices> incomingData;
	const std::array<std::string, fingerCount> fingerNames = { "thumb", "index", "middle", "ring", "pinky" };
	const std::array<std::string, jointPerFingerCount> fingerJointNames = { "mcp", "pip", "dip" };
	const std::array<std::string, jointPerFingerCount> thumbJointNames = { "cmc", "mcp", "ip" };

	HermesSDK::deviceDataCallback onDeviceData = [&](const Hermes::Protocol::Devices& _data) {
		incomingData.enqueue(_data);
		const int numGloves = _data.gloves_size();
		for (int i = 0; i < numGloves; ++i)
		{
			const uint64_t gloveId = _data.gloves()[i].info().deviceid();
			m_timers[gloveId].Log();
			m_timers[clientTimerId].Log();
		}
	};

	Hermes::Protocol::MeshNodeConfig* createMeshNodeConfig(Hermes::Protocol::coor_axis_t _up, Hermes::Protocol::coor_axis_t _forward, Hermes::Protocol::coor_axis_t _right)
	{
		auto meshNodeConfig = new Hermes::Protocol::MeshNodeConfig();
		meshNodeConfig->set_updirection(_up);
		meshNodeConfig->set_forwarddirection(_forward);
		meshNodeConfig->set_rightdirection(_right);
		return meshNodeConfig;
	}

	// MeshConfig defines the coordinate frame in a node/bone of the mesh
	//
	// We assume that hand mesh bones are oriented in such a way that one axis points towards the tips of the fingers,
	// one axis points out of the hand perpendicular to the palmar plane or the plane in which the finger nails would lie (in case of the thumb),
	// and one axis perpendicular to the other two, usually designating the axis of rotation of the finger joints.
	//
	// For the world frame, forward = from viewer, right = right of viewer
	// Some engines flip a coordinate frame axis to convert a right-handed coordinate frame to a left-handed coordinate frame
	// This negation axis should also be specified for your application
	//
	// Example: The bone that you try to describe has its x-Axis pointing towards the finger tip, its y-Axis pointing towards the palm and rotates around the z-Axis, which points left.
	// In this case, upDirection = COOR_AXIS_Y_NEG (the negation of the y-Axis points to the back of the hand), forwardDirection = COOR_AXIS_X_POS,
	// and rightDirection = COOR_AXIS_Z_NEG (the negation of the z-Axis points right according to the above definition)
	//
	// Attention: The upAxis for the thumb refers to the direction outward from the back of the thumb,
	// and its rightAxis refers to the direction pointing right when looking at the back of the thumb (i.e. the side on which the nail is)
	//
	// Note: for an illustration check out the online documentation for the Apollo Network SDK
	// https://docs.google.com/document/d/1JOsypIvvdu783DmdemnJ3TCxMYAw_VZL9HvqoFW55Sc/edit#heading=h.v6d1qnger7bn

	void createMeshConfig(std::string* _bytes)
	{
		auto meshConfig = new Hermes::Protocol::MeshConfig();

		//	Left for our Unity plugin: +Y +X -Z
		auto leftConfig = createMeshNodeConfig(Hermes::Protocol::coor_axis_t::CoorAxisYpos, Hermes::Protocol::coor_axis_t::CoorAxisXpos, Hermes::Protocol::coor_axis_t::CoorAxisZneg);
		meshConfig->set_allocated_leftwrist(leftConfig);
		meshConfig->set_allocated_leftthumb(leftConfig); // note: up-axis is defined as outward of nail, not world up-axis
		meshConfig->set_allocated_leftfinger(leftConfig);

		//	Right for our Unity plugin:: -Y -X -Z
		auto rightConfig = createMeshNodeConfig(Hermes::Protocol::coor_axis_t::CoorAxisYneg, Hermes::Protocol::coor_axis_t::CoorAxisXneg, Hermes::Protocol::coor_axis_t::CoorAxisZneg);
		meshConfig->set_allocated_rightwrist(rightConfig);
		meshConfig->set_allocated_rightthumb(rightConfig); // note: up-axis is defined as outward of nail, not world up-axis
		meshConfig->set_allocated_rightfinger(rightConfig);

		//	World for Unity: +Y +Z +X
		auto worldConfig = createMeshNodeConfig(Hermes::Protocol::coor_axis_t::CoorAxisYpos, Hermes::Protocol::coor_axis_t::CoorAxisZpos, Hermes::Protocol::coor_axis_t::CoorAxisXpos);
		meshConfig->set_allocated_world(worldConfig);

		// Unity uses a left-handed coordinate system and therefore negates the x-axis, this might not be true for your application!
		// https://gamedev.stackexchange.com/questions/39906/why-does-unity-obj-import-flip-my-x-coordinate
		// https://fogbugz.unity3d.com/default.asp?983147_lnn32r51vrk1cpna
		meshConfig->set_negateaxisx(true);
		meshConfig->set_negateaxisy(false);
		meshConfig->set_negateaxisz(false);

		size_t size = meshConfig->ByteSizeLong();
		meshConfig->SerializeToString(_bytes);
	}

	// In this function, the client can request filters and change their parameters
	HermesSDK::filterSetupCallback onFilterSetup = [&]( Hermes::Protocol::Pipeline& _pipeline )
	{
		_pipeline.set_name("C++ sample pipeline");
		_pipeline.clear_filters();

		// For testing, you can add a procedural dummy glove, it will start generating data without any hardware attached.
		//auto proceduralglove = _pipeline.add_filters();
		//proceduralglove->set_name( "ProceduralGlove" );

		// The CreepCompensation-filter compensates the sensor value dropping when the joint is not moving
		// If not added here, the filter will be added by default in ManusCore
		auto CreepCompensationFilter = _pipeline.add_filters();
		CreepCompensationFilter->set_name("CreepCompensation");

		// The NormalizedHand-filter allows to specify the positive direction for finger spreading values
		// If not added here, the filter will be added by default in ManusCore with default parameters
		auto NormalizedHandFilter = _pipeline.add_filters();
		NormalizedHandFilter->set_name("NormalizedHand");

		// Filter parameter "generateHandLocalData" specifies the positive direction of the spread values
		// Set to true:  Finger spread values are positive towards the thumb and negative away from the thumb
		//				  This makes it easier to specify gestures that are generic for left and right hands
		// Set to false: Default because of backwards compatibility. When your flat hands are facing downwards,
		//				  finger spreading towards left is negative and finger spreading to the right is positive.
		auto set = new Hermes::Protocol::ParameterSet();
		auto param = set->add_parameters();
		param->set_name("generateHandLocalData");
		param->set_boolean(true);
		NormalizedHandFilter->set_allocated_parameterset(set);

		// The Basisconversion-filter allows to define the coordinate frame for the finger quaternions (filter is added by default)
		// for more info see comment above the createMeshConfig() method
		auto basisConversion = _pipeline.add_filters();
		basisConversion->set_name("BasisConversion");
		auto basisConversionParamSet = new Hermes::Protocol::ParameterSet();
		auto basisConversionParam = basisConversionParamSet->add_parameters();
		basisConversionParam->set_name("serializedMeshConfig");

		std::string bytes;
		createMeshConfig(&bytes);
		basisConversionParam->set_bytes(bytes);
		basisConversion->set_allocated_parameterset(basisConversionParamSet);	//	NOTE:	Takes ownership of the parameterset here, and doesn't need to be explicitly deleted.

		// quaternion decompression not yet implemented in C++, so disable it for now
		//auto quaternionCompressionFilter = _pipeline.add_filters();			//	Add quaternion compression to the end of the pipeline to reduce network load.
		//quaternionCompressionFilter->set_name( "QuaternionCompression" );

	};

	void clear()
	{
		COORD topLeft = { 0, 0 };
		HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
		CONSOLE_SCREEN_BUFFER_INFO screen;
		DWORD written;

		GetConsoleScreenBufferInfo(console, &screen);
		FillConsoleOutputCharacterA(console, ' ', screen.dwSize.X * screen.dwSize.Y, topLeft, &written);
		FillConsoleOutputAttribute(console, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE, screen.dwSize.X * screen.dwSize.Y, topLeft, &written);
		SetConsoleCursorPosition(console, topLeft);
	};

	void gotoxy(short x, short y)
	{
		COORD pos = { x, y };
		HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleCursorPosition(output, pos);
	};

	HermesSDK::deviceLandscapeCallback onLandscapeData = [&](const Hermes::Protocol::Hardware::DeviceLandscape& _data)
	{
		m_Landscape_mutex.lock();
		m_Landscape = _data;

		Sleep(10);
		clear();

		m_Landscape_mutex.unlock();
	};

	void printLandscapeData()
	{
		m_Landscape_mutex.lock();

		// print the landscape below the data glove info lines
		gotoxy(0, (short)(nrOfKeyInfoLines + nrOfDataGloveInfoLines * m_glovePrintOrder.size()));

		// use the landscape to create a mapping from gloveID to the place where it needs to print that glove's data in the console window
		// recreate the mapping every time the landscape is published, because it could have changed because gloves are added/removed
		size_t pairedDataGloveCounter = 0;
		m_glovePrintOrder.clear();

		for (auto& forestkv : m_Landscape.forest())
		{
			Hermes::Protocol::Hardware::DeviceForest::ForestType forestType = forestkv.second.foresttype();

			std::string forestName = forestkv.first;
			spdlog::info("Service '{}'", forestName);

			for (auto& treekv : forestkv.second.trees())
			{
				// "forest" containing "trees" (dongles) with "leafs" (gloves)
				// We will create an enum field soon, to avoid having to parse strings
				bool isDeviceForest = forestType == Hermes::Protocol::Hardware::DeviceForest::ForestType::DeviceForest_ForestType_DevicesForest;
				bool isHapticsForest = forestType == Hermes::Protocol::Hardware::DeviceForest::ForestType::DeviceForest_ForestType_HapticsForest;

				if (isDeviceForest)
				{
					std::string family = FamilyInfo::FamilyToString(treekv.second.family());
					spdlog::info("\tDongle '{:X}' '{}' '{}', Channel: {}, Family: {}", treekv.first, treekv.second.name(), treekv.second.description(), treekv.second.channel(), family);
				}
				else if (isHapticsForest)
				{
					spdlog::info("\tDongle '{:X}' '{}' '{}'", treekv.first, treekv.second.name(), treekv.second.description());
				}
				else // e.g. TrackerForest or PlayerForest (for Polygon)
				{
					spdlog::info("\t'{}' '{}'", treekv.second.name(), treekv.second.description());
				}

				for (auto& leafkv : treekv.second.leafs())
				{
					auto leaf = leafkv.second;

					// only print paired gloves
					if (leaf.paired())
					{
						// GetLeafInfo(leaf) is a convenience function to get generic glove info (e.g. GetLeafInfo(leaf).BatteryPercentage())
						// so it is not needed to check the glove type (e.g. leaf.primeoneglove.batterypercentage() and leaf.primetwoglove.batterypercentage())
						LeafInfo info = HermesSDK::GetLeafInfo(leaf);

						if (info.DeviceOfType() != LeafInfo::DeviceType::HapticsModule)
						{
							spdlog::info("\t\t- Glove '{}', '{}' {:<5} Battery: {}%, Signal: {}dB, Family: {}", info.Name(), info.Description(), handTypeToString(info.HandOfType()), info.BatteryPercentage(), info.TransmissionStrength(), info.FamilyToString());
							m_glovePrintOrder[leaf.id()] = pairedDataGloveCounter;
							pairedDataGloveCounter++;
						}
						else
						{
							// BatteryPercentage(), TransmissionStrength() and FamilyToString() not yet available for PrimeOne HapticsModule
							spdlog::info("\t\t- Glove '{}', '{:<11}' {:<5} Battery: {}%, Signal: {}dB", info.Name(), info.Description(), handTypeToString(info.HandOfType()), info.BatteryPercentage(), info.TransmissionStrength());
						}
					}
				}
			}
		}
		m_Landscape_mutex.unlock();
	};

	// This callback needs to be added, but can be left empty
	// It can be used to get Polygon data from ManusCore (a virtual human skeleton made from tracker data)
	// https://www.manus-vr.com/polygon
	HermesSDK::polygonDataCallback onPolygonData = [&](const Hermes::Protocol::Polygon::Data& _data)
	{
		
	};

	// This callback needs to be added, but can be left empty
	// It can be used to get the raw tracker data from ManusCore
	HermesSDK::trackingDataCallback onTrackingData = [&](const Hermes::Protocol::TrackerData &_data)
	{
		
	};

	std::string handTypeToString(Hermes::Protocol::HandType handType)
	{
		switch (handType)
		{
		case Hermes::Protocol::HandType::Left:							return "Left";
		case Hermes::Protocol::HandType::Right:							return "Right";
		case Hermes::Protocol::HandType::UnknownChirality:	default:	return "Unknown";
		}
	}

	void printDeviceData(Hermes::Protocol::Devices& dev)
	{
		if (dev.gloves_size() > 0)
		{
			for (int i = 0; i < dev.gloves_size(); i++)
			{
				auto glove = dev.gloves(i);

				if (glove.has_info())
				{
					auto info = glove.info();

					const uint64_t gloveId = info.deviceid();

					std::stringstream stream;
					stream << std::hex << std::uppercase << gloveId;
					std::string gloveIdString("0x" + stream.str());

					m_gloveRates[gloveId] = m_timers[gloveId].GetAverageRate();
					m_gloveGaps[gloveId] = m_timers[gloveId].GetLargestGap();

					std::string rumbleWristStr = " ";
					if (m_rumbleState[info.handtype()].wrist)
						rumbleWristStr = "R";

					std::string handtype = handTypeToString(info.handtype());

					gotoxy(0, (short)(nrOfKeyInfoLines + m_glovePrintOrder[gloveId] * nrOfDataGloveInfoLines));

					spdlog::info("[{}] {:<5} glove, id={:<10}, data rate={:>7}, data gap={:>9}", rumbleWristStr, handtype, gloveIdString, m_gloveRates[gloveId], m_gloveGaps[gloveId]);
				}

				auto raw = glove.raw();
				for (int i = 0; i < raw.flex_size(); i++)
				{
					auto rawfinger = raw.flex(i);
					auto quatfinger = glove.fingers(i);

					// use imus for finger spreading, this is the imu rotation RELATIVE TO THE WRIST IMU and NOT relative to the world
					Hermes::Protocol::Orientation imu;
					int imu_nr = i + 1; // imu(0) = wrist, imu(1) = thumb, imu(2) = index, imu(3) = middle, imu(4) = ring, imu(5) = pinky
					if (raw.imus_size() > imu_nr) // primeOne has 2 imu's (wrist and thumb), primeTwo has 6 imu's (wrist + 5 fingers)
					{
						imu = raw.imus(imu_nr);
					}

					std::string rumbleFingerStr = " ";
					if (m_rumbleState[glove.info().handtype()].finger[i])
						rumbleFingerStr = "R";

					bool thumb = (i == 0);
					const std::array<std::string, jointPerFingerCount>& jointNames = thumb ? thumbJointNames : fingerJointNames;

					// press [TAB] to toggle between printing joint angles as normalized values, degrees or quaternions
					switch (m_jointDataViewMode) {
					case JointAnglePrintMode::Degrees:
					{
						printFingerDegrees(rumbleFingerStr, i, jointNames, quatfinger);
						publishFingerDegrees(rumbleFingerStr, i, jointNames, quatfinger);
						break;
					}
					case JointAnglePrintMode::Quaternions:
					{
						printFingerQuaternion(rumbleFingerStr, i, jointNames, quatfinger);
						publishFingerQuaternion(rumbleFingerStr, i, jointNames, quatfinger);
						break;
					}
					case JointAnglePrintMode::Normalized:
					default:
					{
						printFingerNormalized(rumbleFingerStr, i, jointNames, quatfinger);
						publishFingerNormalized(rumbleFingerStr, i, jointNames, quatfinger);
						break;
					}}
				}
			}
		}
	}

	void printFingerNormalized(std::string& rumbleFingerStr, int& i, const std::array<std::string, jointPerFingerCount>& jointNames, Hermes::Protocol::Finger& quatfinger)
	{
		spdlog::info("[{}][{}]: {:<6} [{:<3} spread: {:>5}, stretch: {:>5}] [{:<3} stretch: {:>5}] [{:<3} stretch: {:>5}]",
			rumbleFingerStr, // shows "R" if rumbling
			i, // index of the finger
			fingerNames[i], // name of the finger
			jointNames[0], // cmc for thumb, mcp for other fingers
			roundFloat(quatfinger.phalanges(0).spread(), 2), // mcp joint finger spreading normalized value
			roundFloat(quatfinger.phalanges(0).stretch(), 2), // mcp joint finger bending normalized value, blended between flex and imu sensors
			jointNames[1], // mcp for thumb, pip for other fingers
			roundFloat(quatfinger.phalanges(1).stretch(), 2), // pip joint finger bending normalized value
			jointNames[2], // ip for thumb, dip for other fingers
			roundFloat(quatfinger.phalanges(2).stretch(), 2)); // dip joint finger bending normalized value, no sensor at dip, same as pip... 
	}

	void printFingerDegrees(std::string& rumbleFingerStr, int& i, const std::array<std::string, jointPerFingerCount>& jointNames, Hermes::Protocol::Finger& quatfinger)
	{
		// \370 for printing degrees symbol
		spdlog::info("[{}][{}]: {:<6} [{:<3} spread: {:>5}\370, stretch: {:>5}\370] [{:<3} stretch: {:>5}\370] [{:<3} stretch: {:>5}\370]",
			rumbleFingerStr, // shows "R" if rumbling
			i, // index of the finger
			fingerNames[i], // name of the finger
			jointNames[0], // cmc for thumb, mcp for other fingers
			roundFloat(quatfinger.phalanges(0).spreaddegrees(), 0), // mcp joint finger spreading degrees value
			roundFloat(quatfinger.phalanges(0).stretchdegrees(), 0), // mcp joint finger bending degrees value, blended between flex and imu sensors
			jointNames[1], // mcp for thumb, pip for other fingers
			roundFloat(quatfinger.phalanges(1).stretchdegrees(), 0), // pip joint finger bending degrees value
			jointNames[2], // ip for thumb, dip for other fingers
			roundFloat(quatfinger.phalanges(2).stretchdegrees(), 0)); // dip joint finger bending degrees value, same normalized value as pip, but mapped on different range
	}

	void printFingerQuaternion(std::string& rumbleFingerStr, int& i, const std::array<std::string, jointPerFingerCount>& jointNames, Hermes::Protocol::Finger& quatfinger)
	{
		spdlog::info("[{}][{}]: {:<6} [{:<3} quat x: {:>5} y: {:>5} z: {:>5} w: {:>5}] [{:<3} quat x: {:>5} y: {:>5} z: {:>5} w: {:>5}] [{:<3} quat x: {:>5} y: {:>5} z: {:>5} w: {:>5}]",
			rumbleFingerStr, // shows "R" if rumbling
			i, // index of the finger
			fingerNames[i], // name of the finger
			jointNames[0], // cmc for thumb, mcp for other fingers
			roundFloat(quatfinger.phalanges(0).rotation().full().x(), 2), // joint quaternion X
			roundFloat(quatfinger.phalanges(0).rotation().full().y(), 2), // joint quaternion Y
			roundFloat(quatfinger.phalanges(0).rotation().full().z(), 2), // joint quaternion Z
			roundFloat(quatfinger.phalanges(0).rotation().full().w(), 2), // joint quaternion W
			jointNames[1], // mcp for thumb, pip for other fingers
			roundFloat(quatfinger.phalanges(1).rotation().full().x(), 2), // joint quaternion X
			roundFloat(quatfinger.phalanges(1).rotation().full().y(), 2), // joint quaternion Y
			roundFloat(quatfinger.phalanges(1).rotation().full().z(), 2), // joint quaternion Z
			roundFloat(quatfinger.phalanges(1).rotation().full().w(), 2), // joint quaternion W
			jointNames[2], // ip for thumb, dip for other fingers
			// dip joint finger bending normalized value, no sensor at dip, same as pip
			// but dip quaternion tuned in handmodel is different from pip to make it look natural;
			roundFloat(quatfinger.phalanges(2).rotation().full().x(), 2), // joint quaternion X
			roundFloat(quatfinger.phalanges(2).rotation().full().y(), 2), // joint quaternion Y
			roundFloat(quatfinger.phalanges(2).rotation().full().z(), 2), // joint quaternion Z
			roundFloat(quatfinger.phalanges(2).rotation().full().w(), 2));// joint quaternion W
	}

	HermesSDK::errorMessageCallback onError = [&](const HermesSDK::ErrorMessage& _msg)
	{
		spdlog::error("onError: {}", _msg.errorMessage);
	};

	// round float to _places_ number of places behind floating point
	float roundFloat(float val, int places)
	{
		return std::round(val * std::pow(10, places)) / std::pow(10, places);
	}

	void maximizeWindow()
	{
		try
		{
			HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
			bool absolute = true;
			COORD maxWindowCoord = GetLargestConsoleWindowSize(console);
			SetConsoleScreenBufferSize(console, maxWindowCoord);
			_SMALL_RECT maxWindowRect = { 0, 0, maxWindowCoord.X - 1, maxWindowCoord.Y - 1 };
			SetConsoleWindowInfo(console, absolute, &maxWindowRect);
			HWND consoleWindow = GetConsoleWindow();
			SetWindowPos(consoleWindow, 0, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
		}
		catch (const std::exception & ex)
		{
			spdlog::error("Exception while maximizing window: {}", ex.what());
		}
	}

public:
	Sample(): nh("")
	{
		maximizeWindow();
		pub_norm = nh.advertise<std_msgs::Float32MultiArray>("norm", 10);
		pub_deg = nh.advertise<std_msgs::Float32MultiArray>("degree", 10);
		pub_quat = nh.advertise<std_msgs::Float32MultiArray>("quaternion", 10);
		// sub = nh.subscribe("/chatter", 1, &RosWithClass::Callback, this)
	}

	void publishFingerNormalized(std::string& rumbleFingerStr, int& i, const std::array<std::string, jointPerFingerCount>& jointNames, Hermes::Protocol::Finger& quatfinger)
	{
		std_msgs::Float32MultiArray array;	
		array.data.resize(5);
		array.data[0] = i; // finger number
		array.data[1] = roundFloat(quatfinger.phalanges(0).spread(), 2);
		array.data[2] = roundFloat(quatfinger.phalanges(0).stretch(), 2);
		array.data[3] = roundFloat(quatfinger.phalanges(1).stretch(), 2);
		array.data[4] = roundFloat(quatfinger.phalanges(2).stretch(), 2);
		pub_deg.publish(array);
	}

	void publishFingerDegrees(std::string& rumbleFingerStr, int& i, const std::array<std::string, jointPerFingerCount>& jointNames, Hermes::Protocol::Finger& quatfinger)
	{
		std_msgs::Float32MultiArray array;
		array.data.resize(5);
		array.data[0] = i; // finger number
		array.data[1] = roundFloat(quatfinger.phalanges(0).spreaddegrees(), 0);
		array.data[2] = roundFloat(quatfinger.phalanges(0).stretchdegrees(), 0);
		array.data[3] = roundFloat(quatfinger.phalanges(1).stretchdegrees(), 0);
		array.data[4] = roundFloat(quatfinger.phalanges(2).stretchdegrees(), 0);
		pub_deg.publish(array);
		// This part publishes each degree of finger joints.
		// Be careful to use. 
	}

	void publishFingerQuaternion(std::string& rumbleFingerStr, int& i, const std::array<std::string, jointPerFingerCount>& jointNames, Hermes::Protocol::Finger& quatfinger)
	{
		std_msgs::Float32MultiArray quat;
		quat.data.resize(13);
		quat.data[0] = i; // finger number
		quat.data[1] = roundFloat(quatfinger.phalanges(0).rotation().full().x(), 2);
		quat.data[2] = roundFloat(quatfinger.phalanges(0).rotation().full().y(), 2);
		quat.data[3] = roundFloat(quatfinger.phalanges(0).rotation().full().z(), 2);
		quat.data[4] = roundFloat(quatfinger.phalanges(0).rotation().full().w(), 2);
		quat.data[5] = roundFloat(quatfinger.phalanges(1).rotation().full().x(), 2);
		quat.data[6] = roundFloat(quatfinger.phalanges(1).rotation().full().y(), 2);
		quat.data[7] = roundFloat(quatfinger.phalanges(1).rotation().full().z(), 2);
		quat.data[8] = roundFloat(quatfinger.phalanges(1).rotation().full().w(), 2);
		quat.data[9] = roundFloat(quatfinger.phalanges(2).rotation().full().x(), 2);
		quat.data[10] = roundFloat(quatfinger.phalanges(2).rotation().full().y(), 2);
		quat.data[11] = roundFloat(quatfinger.phalanges(2).rotation().full().z(), 2);
		quat.data[12] = roundFloat(quatfinger.phalanges(2).rotation().full().w(), 2);
		pub_quat.publish(quat);
		// This part publishes each degree of finger joints.
		// Be careful to use. 
	}


	/// <summary>
	/// connect to a local host on this pc. make sure it is active before trying!
	/// </summary>
	/// <param name="_clientName"></param>
	/// <param name="_clientInfo"></param>
	/// <returns></returns>
	bool	ConnectLocal(const std::string& _clientName, const std::string& _clientInfo)
	{
		HermesSDK::ConnectLocal(_clientName, _clientInfo, this->onFilterSetup, this->onDeviceData, this->onLandscapeData, this->onPolygonData, this->onTrackingData, this->onError );

		m_timeLastHapticsCmdSent = std::chrono::high_resolution_clock::now(); // initialize the haptics clock

		return true;
	}

	/// <summary>
	/// connect to a host based on its hostname.
	/// </summary>
	/// <param name="_clientName"></param>
	/// <param name="_clientInfo"></param>
	/// <param name="_hostName"> the hostname you want to connect to</param>
	/// <returns></returns>
	bool	ConnectNetworkByHostName(const std::string& _clientName, const std::string& _clientInfo, const std::string& _hostName)
	{
		HermesSDK::ConnectNetworkHostName(_clientName, _clientInfo, _hostName, this->onFilterSetup, this->onDeviceData, this->onLandscapeData, this->onPolygonData, this->onTrackingData, this->onError);

		m_timeLastHapticsCmdSent = std::chrono::high_resolution_clock::now(); // initialize the haptics clock

		return true;
	}

	/// <summary>
	/// connect to a host based on its (ip) address.
	/// </summary>
	/// <param name="_clientName"></param>
	/// <param name="_clientInfo"></param>
	/// <param name="_address">the (ip) address you want to connect to</param>
	/// <returns></returns>
	bool	ConnectNetworkByAddress(const std::string& _clientName, const std::string& _clientInfo, const std::string& _address)
	{
		HermesSDK::ConnectNetworkAddress(_clientName, _clientInfo, _address, this->onFilterSetup, this->onDeviceData, this->onLandscapeData, this->onPolygonData, this->onTrackingData, this->onError);

		m_timeLastHapticsCmdSent = std::chrono::high_resolution_clock::now(); // initialize the haptics clock

		return true;
	}

	/// <summary>
	/// finds all the manus core (hermes) hosts on the network
	/// do NOT use this during a connection with a host as it will conflict with any current connection.
	/// use it beforehand to get a list of hostnames and addresses so you know there is a connection possible, or to test if your network is
	/// cooperating. if your desired (or any ) host is not found, you want to check your network settings! (the firewall probably needs an exclusion)
	/// for more complex networks (segmented etc) port forwarding etc must be handled in the network routers/switches/bridges/servers and is out of scope for this SDK.(check windows)
	/// </summary>
	/// <returns>returns a map of their hostnames and addresses </returns>
	std::map<std::string, std::string> findHosts()
	{
		HermesSDK::FindNetworkHosts(this->onFilterSetup, this->onDeviceData, this->onLandscapeData, this->onPolygonData, this->onTrackingData, this->onError);

		m_timeLastHapticsCmdSent = std::chrono::high_resolution_clock::now(); // initialize the haptics clock

		std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // lets scan for 2 seconds (magic number) for all the hosts.
		// get the list.
		std::map<std::string, std::string> hosts = HermesSDK::GetFoundHosts();
		HermesSDK::Stop(); // and then stop.
		return hosts;
	}

	bool Disconnect()
	{
		HermesSDK::Stop();

		for (const auto& timer : m_timers)
		{
			std::string filename;

			if (timer.first == clientTimerId) {
				filename = "_dataRateLog_@_Client";
			}
			else {
				std::stringstream stream;
				stream << std::hex << std::uppercase << timer.first;
				filename = "_dataRateLog_@_Client_Glove_0x" + stream.str();
			}

			timer.second.WriteLog(filename + ".csv");
		}

		return true;
	}

	bool key_pressed(int key)
	{
		return (GetAsyncKeyState(key) & 0x8000);
	}

	void handleRumbleCommands()
	{
		m_rumbleState[Hermes::Protocol::Left].wrist = key_pressed(VK_LEFT);
		m_rumbleState[Hermes::Protocol::Right].wrist = key_pressed(VK_RIGHT);

		// strange key number sequence results from having gloves lie in front of you, and have the keys and rumblers in the same order
		m_rumbleState[Hermes::Protocol::Left].finger[0] = key_pressed('5'); // left thumb
		m_rumbleState[Hermes::Protocol::Left].finger[1] = key_pressed('4'); // left index
		m_rumbleState[Hermes::Protocol::Left].finger[2] = key_pressed('3'); // left middle
		m_rumbleState[Hermes::Protocol::Left].finger[3] = key_pressed('2'); // left ring
		m_rumbleState[Hermes::Protocol::Left].finger[4] = key_pressed('1'); // left pinky
		m_rumbleState[Hermes::Protocol::Right].finger[0] = key_pressed('6'); // right thumb
		m_rumbleState[Hermes::Protocol::Right].finger[1] = key_pressed('7'); // right index
		m_rumbleState[Hermes::Protocol::Right].finger[2] = key_pressed('8'); // right middle
		m_rumbleState[Hermes::Protocol::Right].finger[3] = key_pressed('9'); // right ring
		m_rumbleState[Hermes::Protocol::Right].finger[4] = key_pressed('0'); // right pinky

		unsigned long long elapsedLastCmd_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_timeLastHapticsCmdSent).count();

		if (elapsedLastCmd_ms > timeBetweenHapticsCmds_ms) {
			m_timeLastHapticsCmdSent = std::chrono::high_resolution_clock::now();

			m_hapticsTimer.Log();

			m_Landscape_mutex.lock();
			for (auto& forestkv : m_Landscape.forest()) {
				for (auto& treekv : forestkv.second.trees()) {
					auto tree = treekv.second;
					for (auto& leafkv : treekv.second.leafs()) {
						auto leaf = leafkv.second;
						if (leaf.id() != 0 && leaf.paired()) {

							float full_power = 1.0f; // full power
							Hermes::Protocol::HandType handtype = HermesSDK::GetLeafInfo(leaf).HandOfType();

							// Wrist vibration for PrimeOne and PrimeTwo is currently not supported and disabled
							/*
							if (leaf.has_primetwoglove() || leaf.has_primeoneglove() || leaf.has_apollolegacyglove())
							{
								// Rumble the wrist
								bool success = HermesSDK::VibrateWrist(leaf.id(), full_power * m_rumbleState[handtype].wrist, 0);
							}
							*/

							// Finger vibration only works for PrimeOne Haptics and PrimeTwo Haptics gloves
							if (HermesSDK::GetLeafInfo(leaf).HasHaptics())
							{
								// Rumble the fingers
								std::array<float, fingerCount> rumble_powers;
								for (int i = 0; i < fingerCount; ++i) {
									rumble_powers[i] = full_power * m_rumbleState[handtype].finger[i];
								}

								bool success = HermesSDK::VibrateFingers(tree.id(), handtype, rumble_powers);
							}
						}
					}
				}
			}

			m_Landscape_mutex.unlock();
		}
	}

	void handleRumbleCallback(const std_msgs::Int16& msg)
	{
		/* if (msg.data == 5) {
			m_rumbleState[Hermes::Protocol::Left].finger[0] = key_pressed('5'); // left thumb
		}
		if (msg.data == 4) {
			m_rumbleState[Hermes::Protocol::Left].finger[1] = key_pressed('4'); // left index
		}
		if (msg.data == 3) {
			m_rumbleState[Hermes::Protocol::Left].finger[2] = key_pressed('3'); // left middle
		}
		if (msg.data == 2) {
			m_rumbleState[Hermes::Protocol::Left].finger[3] = key_pressed('2'); // left ring
		}
		if (msg.data == 1) {
			m_rumbleState[Hermes::Protocol::Left].finger[0] = key_pressed('5'); // left pinky
		}
		if (msg.data == 6) {
			m_rumbleState[Hermes::Protocol::Right].finger[0] = key_pressed('6'); // right thumb
		}
		if (msg.data == 7) {
			m_rumbleState[Hermes::Protocol::Right].finger[1] = key_pressed('7'); // right index
		}
		if (msg.data == 8) {
			m_rumbleState[Hermes::Protocol::Right].finger[2] = key_pressed('8'); // right middle
		}
		if (msg.data == 9) {
			m_rumbleState[Hermes::Protocol::Right].finger[3] = key_pressed('9'); // right ring
		}
		if (msg.data == 0) {
			m_rumbleState[Hermes::Protocol::Right].finger[4] = key_pressed('0'); // right pinky
		} */
		// strange key number sequence results from having gloves lie in front of you, and have the keys and rumblers in the same order

		unsigned long long elapsedLastCmd_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_timeLastHapticsCmdSent).count();

		if (elapsedLastCmd_ms > timeBetweenHapticsCmds_ms) {
			m_timeLastHapticsCmdSent = std::chrono::high_resolution_clock::now();

			m_hapticsTimer.Log();

			m_Landscape_mutex.lock();
			for (auto& forestkv : m_Landscape.forest()) {
				for (auto& treekv : forestkv.second.trees()) {
					auto tree = treekv.second;
					for (auto& leafkv : treekv.second.leafs()) {
						auto leaf = leafkv.second;
						if (leaf.id() != 0 && leaf.paired()) {

							float full_power = 1.0f; // full power
							Hermes::Protocol::HandType handtype = HermesSDK::GetLeafInfo(leaf).HandOfType();

							// Wrist vibration for PrimeOne and PrimeTwo is currently not supported and disabled
							/*
							if (leaf.has_primetwoglove() || leaf.has_primeoneglove() || leaf.has_apollolegacyglove())
							{
								// Rumble the wrist
								bool success = HermesSDK::VibrateWrist(leaf.id(), full_power * m_rumbleState[handtype].wrist, 0);
							}
							*/

							// Finger vibration only works for PrimeOne Haptics and PrimeTwo Haptics gloves
							if (HermesSDK::GetLeafInfo(leaf).HasHaptics())
							{
								// Rumble the fingers
								std::array<float, fingerCount> rumble_powers;
								for (int i = 0; i < fingerCount; ++i) {
									rumble_powers[i] = full_power * m_rumbleState[handtype].finger[i];
								}

								bool success = HermesSDK::VibrateFingers(tree.id(), handtype, rumble_powers);
							}
						}
					}
				}
			}

			m_Landscape_mutex.unlock();
		}
	}

	// return 'false' to keep running, return 'true' to request exit
	bool	Update()
	{
		bool requestExit = false;

		if (key_pressed(VK_ESCAPE))
		{
			spdlog::info("Pressed escape...");
			requestExit = true;
		}

		if (key_pressed(VK_TAB)) // print quaternions, degrees or normalized values		
		{
			m_tabPreviouslyPressed = true; // remember the pressed state, so we can detect a tab release
		}
		else
		{
			if (m_tabPreviouslyPressed) // on tab release (it was pressed in previous update, and now it isn't)
			{
				// cycle through enum values
				m_jointDataViewMode = static_cast<JointAnglePrintMode>((static_cast<int>(m_jointDataViewMode) + 1) % static_cast<int>(JointAnglePrintMode::NumberOfAnglePrintModes));
			}
			m_tabPreviouslyPressed = false;
		}

		if (!HermesSDK::IsRunning())
		{
			// wait until a local hive node is found
			m_Print_mutex.lock();
			gotoxy(0, 0);
			clear();
			spdlog::info("Waiting for user to start ManusCore...");
			m_Print_mutex.unlock();
			return requestExit;
		}

		// handleRumbleCommands();
		// ros::NodeHandle n;
		// ros::Subscriber sub;
		//sub = n.subscribe("rumble", 10, handleRumbleCallback);

		if (incomingData.size_approx() == 0)
		{
			m_updates_without_data++;
			if (m_updates_without_data > 10) // if there is no data for just a single update, we can ignore it
			{
				// inform user that gloves are probably not connected and turned on
				m_Print_mutex.lock();
				gotoxy(0, 0);
				clear();
				spdlog::info("Waiting for user to connect and turn on gloves...");
				m_Print_mutex.unlock();
			}
			return requestExit;
		}
		m_updates_without_data = 0; // if we are here, we have data, and can reset the counter

		Hermes::Protocol::Devices dev;

		incomingData.try_dequeue(dev);

		// get the latest data, discard the rest
		while (incomingData.try_dequeue(dev))
		{
		}

		m_Print_mutex.lock();
		gotoxy(0, 0);

		spdlog::info("[TAB]   = print joint angles as normalized values, degrees or quaternions, [ESC] = quit");
		spdlog::info("[1]-[5] = pinky-thumb rumble keys left  haptics glove");
		spdlog::info("[6]-[0] = thumb-pinky rumble keys right haptics glove");

		printDeviceData(dev);
		printLandscapeData();
		m_Print_mutex.unlock();

		return requestExit;
	}

	void printHosts(std::map<std::string, std::string>& _hosts)
	{
		spdlog::info("-Found Hosts at: (Hostname / Address)");
		std::map<std::string, std::string>::iterator t_It;
		for (t_It = _hosts.begin(); t_It != _hosts.end(); t_It++)
		{
			spdlog::info(" " + t_It->first + " / " + t_It->second);
		}
		spdlog::info("-");

		spdlog::info("Continuing in:");
		for (int i = 3; i > 0; i--)
		{
			spdlog::info(i);
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}
};

// Sample::Sample()
// {
// 	maximizeWindow();
// 	//pub_deg = nh.advertise("/array", 1);
// 	//pub_quat = nh.advertise("/quat", 1);
// 	//sub = nh.subscribe("/chatter", 1, &chatter_cb, this);
// }

void chatter_cb(const std_msgs::Int16& msg)
{
	Sample sample;
	sample.Update();
}

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "manus_glove");
	// ros::NodeHandle nh;
	// ros::Publisher pub_norm;
	// ros::Publisher pub_deg;
	// ros::Publisher pub_quat;
	// ros::Subscriber sub;
	// std_msgs::Float32MultiArray array;
	// std_msgs::Float32MultiArray quat;
	/* ros::NodeHandle n2;
	ros::Subscriber sub = n2.subscribe("chatter", 10, chatter_cb); */
	/* ros::init(argc, argv, "talker");
	ros::NodeHandle n;
	ros::Publisher pub = n.advertise<std_msgs::Float32MultiArray>("array",10);
	ros::Rate loop_rate(1);
	while (ros::ok())
	{
		std_msgs::Float32MultiArray array;
		array.data.resize(4);
		array.data[0] = 0.0;
		array.data[1] = 1.0;
		array.data[2] = 2.0;
		array.data[3] = 3.0;
		pub.publish(array);
		ROS_INFO("I published array!");
		ros::spinOnce();
		loop_rate.sleep();
	} */

	GOOGLE_PROTOBUF_VERIFY_VERSION;

	Sample sample;

	bool requestDisconnect = false;

	while (true)
	{
		// Uncomment this to connect to a remote host on the network
		// std::map<std::string, std::string> t_Hosts = sample.findHosts(); // get a map of found hostnames and addresses
		// sample.printHosts(t_Hosts); // print all found hosts with their addresses
		// sample.ConnectNetworkByHostName("SdkSample", "C++ sample for the Hermes sdk", t_Hosts.begin()->first); // connect by hostname
		// sample.ConnectNetworkByAddress( "SdkSample", "C++ sample for the Hermes sdk", t_Hosts.begin()->second); // connect by address

		// For now we will connect to a local host on this pc
		sample.ConnectLocal("SdkSample", "C++ sample for the Hermes sdk");

		requestDisconnect = false;
		while (!requestDisconnect)
		{
			requestDisconnect = sample.Update();
			ros::NodeHandle n2;
			ros::Subscriber sub = n2.subscribe("chatter", 10, chatter_cb);
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		sample.Disconnect();

		spdlog::info("Press enter to connect again...");
		std::cin.get();
	}
}
