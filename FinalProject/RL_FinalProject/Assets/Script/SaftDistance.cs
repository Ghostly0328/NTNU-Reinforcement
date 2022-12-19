using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class SaftDistance : Agent
{
    [SerializeField] private float currentSpeed;
    [SerializeField] private Transform BusTrans;
    [SerializeField] private Transform Goal;
    [SerializeField] private float distance;

    [SerializeField] private bool isInGoal;
    [SerializeField] private int goalCount;

    [Header("Camera")]
    [SerializeField] private Camera vehicleCam;
    [SerializeField] private Transform YOLODetectLD;
    [SerializeField] private Transform YOLODetectRU;

    [SerializeField] private float moveSpeed = 1f;
    [SerializeField] private Transform targetTransform;
    [SerializeField] private Material winMaterial;
    [SerializeField] private Material loseMaterial;
    [SerializeField] private MeshRenderer floorMeshRenderer;


    public override void OnEpisodeBegin()
    {
        currentSpeed = Random.Range(40, 130); // 40, 130
        float currentSpeedPerSecond = currentSpeed * 1000 / 3600;

        // Set transform
        transform.localPosition = new Vector3(Random.Range(-85, -10), 0, 0); // -85, -10
        BusTrans.localPosition = new Vector3(Random.Range(10, 80), 0, 0);

        distance = Function(currentSpeedPerSecond);
        Goal.localPosition = new Vector3(0, 0, -1 * distance);

        isInGoal = false;
        goalCount = 0;
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        //sensor.AddObservation(transform.position);
        //sensor.AddObservation(targetTransform.position);

        // Cal the Area and Dist
        Vector2 positionLD = vehicleCam.WorldToScreenPoint(YOLODetectLD.position);
        Vector2 positionRU = vehicleCam.WorldToScreenPoint(YOLODetectRU.position);

        //positionLD = positionLD / new Vector2(vehicleCam.pixelWidth, vehicleCam.pixelHeight);
        //positionRU = positionRU / new Vector2(vehicleCam.pixelWidth, vehicleCam.pixelHeight);

        float area = Mathf.Abs((positionRU.x - positionLD.x) * (positionRU.y - positionLD.y));

        //float distance = GetDistance();
        sensor.AddObservation(positionLD.y);
        sensor.AddObservation(distance);
        sensor.AddObservation(area);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Move The Car
        float moveX = actions.ContinuousActions[0];
        transform.position += new Vector3(moveX, 0, 0) * Time.deltaTime * moveSpeed;

        // Add Reward
        if (isInGoal)
        {
            goalCount++;
            AddReward(0.005f);
        }
        else
        {
            AddReward(-100 / MaxStep);
        }
        if (goalCount > 200)
        {
            SetReward(2f);
            floorMeshRenderer.material = winMaterial;
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> contionousActions = actionsOut.ContinuousActions;
        contionousActions[0] = Input.GetAxisRaw("Horizontal");
    }


    private float GetDistance()
    {
        return Vector3.Distance(YOLODetectLD.position, vehicleCam.gameObject.transform.position);
    }

    private float Function(float currentSpeed)
    {
        return (float)(currentSpeed * currentSpeed / (2 * 9.8 * 0.8));
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Wall>(out Wall wall))
        {
            SetReward(-1f);
            floorMeshRenderer.material = loseMaterial;
            EndEpisode();
        }
    }
    private void OnTriggerStay(Collider other)
    {
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            isInGoal = true;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            isInGoal = false;
        }
    }
}
