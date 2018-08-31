/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#ifndef __GAZEBO_ARM_PLUGIN_H__
#define __GAZEBO_ARM_PLUGIN_H__

#include "deepRL.h"

#include <boost/bind.hpp>
#include <gazebo/gazebo.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <gazebo/transport/TransportTypes.hh>
#include <gazebo/msgs/MessageTypes.hh>
#include <gazebo/common/Time.hh>

#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>
#include <syslog.h>
#include <time.h>


namespace gazebo
{

/**
 * ArmPlugin
 */
class ArmPlugin : public ModelPlugin
{
public: 
	ArmPlugin(); 

	virtual void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/); 
	virtual void OnUpdate(const common::UpdateInfo & /*_info*/);

	float resetPosition( uint32_t dof );  // center servo positions

	bool createAgent();
	bool updateAgent();
	bool updateJoints();

	void onCameraMsg(ConstImageStampedPtr &_msg);
	void onCollisionMsg(ConstContactsPtr &contacts);

	static const uint32_t DOF  = 2;	// active degrees of freedom in the arm

private:
	float ref[DOF];			// joint reference positions
	float vel[DOF];			// joint velocity control
	float dT[3];				// IK delta theta

	// rlAgent* agent;			// AI learning agent instance
	dqnAgent* agent;

	bool     newState;			// true if a new frame needs processed
	bool     newReward;			// true if a new reward's been issued
	bool     endEpisode;		// true if this episode is over
	float    rewardHistory;		// value of the last reward issued
	Tensor*  inputState;		// pyTorch input object to the agent
	void*    inputBuffer[2];		// [0] for CPU and [1] for GPU
	size_t   inputBufferSize;
	size_t   inputRawWidth;
	size_t   inputRawHeight;	
	float    jointRange[DOF][2];	// min/max range of each arm joint
	float    actionJointDelta;	// amount of offset caused to a joint by an action
	float    actionVelDelta;		// amount of velocity offset caused to a joint by an action
	int	    maxEpisodeLength;	// maximum number of frames to win episode (or <= 0 for unlimited)
	int      episodeFrames;		// frame counter for the current episode	
	bool     testAnimation;		// true for test animation mode
	bool     loopAnimation;		// loop the test animation while true
	uint32_t animationStep;
	float    resetPos[DOF];
	float    lastGoalDistance;
	float    avgGoalDelta;
	float 	 lastDelta;			//store previous delta
	int	     successfulGrabs; 	
	
	bool 	wasCollision;		// flag set if prop-arm collision occurs(including gripper)
	int 	totalCollision;		// to record total prop-arm collisions(including gripper)

	bool 	wasGripperCollision;	// flag set if gripper-prop collision occurs
	int 	gripperCollision;	// to record total gripper-arm collisions

	int	    totalRuns;			// to record total number of episodes
	
	int      runHistoryIdx;		
	int	    runHistoryMax;
	bool     runHistory[20];
	
	std::ofstream outfile;		// to write to file for gnuplots

	physics::ModelPtr model;
	event::ConnectionPtr updateConnection;
	physics::JointController* j2_controller;

	gazebo::transport::NodePtr cameraNode;
	gazebo::transport::SubscriberPtr cameraSub;

	gazebo::transport::NodePtr collisionNode;
	gazebo::transport::SubscriberPtr collisionSub;
};

}


#endif

